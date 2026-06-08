use std::collections::HashMap;

use crate::{
    Floating,
    context::Context,
    graph::Graph,
    identity::Id,
    ops::{Add, Const, Sum},
    tracing::TensorData,
};

#[derive(Debug, Clone)]
pub struct TraceableFn<D: Floating> {
    pub graph: Graph<D>,
    pub inputs: Vec<Id>,
    pub outputs: Vec<Id>,
}

pub trait EvalArgs<D: Floating> {
    fn pack(self) -> Vec<TensorData<D>>;
}

impl<D: Floating + 'static> TraceableFn<D> {
    fn run<T: EvalArgs<D>, O: EvalOutputs<D>>(&self, args: T) -> O {
        let packed = args.pack();
        let mut ctx = Context::<D>::new();

        for (id, val) in self.inputs.iter().zip(packed.into_iter()) {
            ctx.insert(*id, val);
        }

        for op in &self.graph.nodes {
            op.eval(&mut ctx);
        }

        O::from_vec(
            self.outputs
                .iter()
                .map(|id| ctx.checked_get(id).clone())
                .collect(),
        )
    }

    pub fn eval<T, O>(&self) -> impl Fn(T) -> O
    where
        T: EvalArgs<D>,
        O: EvalOutputs<D>,
    {
        move |args: T| self.run(args)
    }

    pub fn grad(&self) -> Self {
        let mut g = self.graph.clone();

        let mut final_output_id = *self
            .outputs
            .first()
            .expect("Cannot differentiate a function with no outputs");

        if self.outputs.len() > 1 {
            for &output_id in self.outputs.iter().skip(1) {
                let new_sum_id = g.fresh();
                g.push(Box::new(Add::new(final_output_id, output_id, new_sum_id)));
                final_output_id = new_sum_id;
            }
        }

        let scalar_output_id = g.fresh();
        g.push(Box::new(Sum::new(
            final_output_id,
            scalar_output_id,
            vec![],
            false,
        )));

        let mut gradients: HashMap<Id, Id> = HashMap::new();
        let seed = g.fresh();
        g.push(Const::boxed(D::one(), seed));
        gradients.insert(scalar_output_id, seed);

        let forward_nodes = std::mem::take(&mut g.nodes);

        for node in forward_nodes.iter().rev() {
            let out_ids = node.outputs();
            let out_grads: Vec<_> = out_ids
                .iter()
                .filter_map(|out| gradients.get(out).copied())
                .collect();

            if out_grads.is_empty() {
                continue;
            }

            // This is now valid because the loop isn't borrowing `g`.
            if let Some(inp_grad) = node.vjp(&mut g, &out_grads) {
                for (inp, grad_contrib) in node.inputs().into_iter().zip(inp_grad) {
                    if let Some(existing) = gradients.get(&inp).copied() {
                        let out = g.fresh();
                        g.push(Box::new(Add::new(existing, grad_contrib, out)));
                        gradients.insert(inp, out);
                    } else {
                        gradients.insert(inp, grad_contrib);
                    }
                }
            }
        }

        let mut vjp_nodes = g.nodes;
        g.nodes = forward_nodes;
        g.nodes.append(&mut vjp_nodes);

        let grads_out: Vec<_> = self
            .inputs
            .iter()
            .map(|i| {
                gradients.get(i).copied().unwrap_or_else(|| {
                    let z = g.fresh();
                    g.push(Box::new(Const::new(D::zero(), z)));
                    z
                })
            })
            .collect();

        Self {
            graph: g,
            inputs: self.inputs.clone(),
            outputs: grads_out,
        }
    }
}

pub trait EvalOutputs<D> {
    fn from_vec(f: Vec<TensorData<D>>) -> Self;
}

use ndarray::{ArrayBase, Data, Dimension};

pub trait ToTensorData<D: Floating> {
    fn to_tensor(&self) -> TensorData<D>;
}

impl<D: Floating, S, Dim> ToTensorData<D> for ArrayBase<S, Dim>
where
    S: Data<Elem = D>,
    Dim: Dimension,
{
    fn to_tensor(&self) -> TensorData<D> {
        self.to_owned().into_dyn()
    }
}

mod macros {
    use super::{EvalArgs, EvalOutputs, Floating, TensorData};
    macro_rules! as_owned_ty {
        ($_:ident, $D:ident) => {
            TensorData<$D>
        };
    }

    #[allow(unused)]
    macro_rules! as_ref_ty {
        ($_:ident, $D:ident) => {
            &TensorData<$D>
        };
    }

    macro_rules! reverse_order {
    // base case
    (($last:ident), $vec:ident) => {
        let $last = $vec.pop().unwrap();
    };

    // recursive case
    (($first:ident, $($rest:ident),+), $vec:ident) => {
        reverse_order!(($($rest),+), $vec);
        let $first = $vec.pop().unwrap();
    };
    }

    macro_rules! impl_eval_outputs {
        ( $( $len:literal => ( $( $name:ident ),+ ) ),+ $(,)? ) => {
            $(
                // #[expect(unused_parens, reason = "macro complains")]
                impl<D: Floating> EvalOutputs<D>
                for ( $( as_owned_ty!($name, D) ),+ ,) {
                    fn from_vec(mut f: Vec<TensorData<D>>) -> Self {
                        assert_eq!(f.len(), $len);
                        reverse_order!(($($name),+), f);
                        ( $( $name ),+ ,)
                    }
                }
            )+
        };
    }

    macro_rules! impl_eval_args {
    ( $( $len:literal => ( $( $name:ident ),+ ) ),+ $(,)? ) => {
        $(
            #[allow(unused_parens, non_camel_case_types)]
            impl<D, $( $name ),+> EvalArgs<D> for ( $( &$name ),+ )
            where
                D: Floating,
                $( $name: $crate::tracing::function::ToTensorData<D> ),+
            {
                fn pack(self) -> Vec<TensorData<D>> {
                    let ( $( $name ),+ ) = self;
                    vec![ $( $name.to_tensor() ),+ ]
                }
            }
        )+
    };
}

    impl_eval_args! {
        1  => (a),
        2  => (a,b),
        3  => (a,b,c),
        4  => (a,b,c,d),
        5  => (a,b,c,d,e),
        6  => (a,b,c,d,e,f),
        7  => (a,b,c,d,e,f,g),
        8  => (a,b,c,d,e,f,g,h),
        9  => (a,b,c,d,e,f,g,h,i),
       10  => (a,b,c,d,e,f,g,h,i,j),
    }

    impl_eval_outputs! {
        1  => (a),
        2  => (a,b),
        3  => (a,b,c),
        4  => (a,b,c,d),
        5  => (a,b,c,d,e),
        6  => (a,b,c,d,e,f),
        7  => (a,b,c,d,e,f,g),
        8  => (a,b,c,d,e,f,g,h),
        9  => (a,b,c,d,e,f,g,h,i),
       10  => (a,b,c,d,e,f,g,h,i,j),
    }
}
