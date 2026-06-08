use itertools::Itertools as _;
use ndarray::Axis;

use crate::{
    Floating, Graph, Id, TraceSession, Tracer,
    context::Context,
    ops::{Op, broadcast::BroadcastLike},
};

#[derive(Debug, Clone)]
pub struct Sum {
    inp: Id,
    out: Id,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl Sum {
    pub fn new(inp: Id, out: Id, axis: impl Into<Vec<usize>>, keep_dims: bool) -> Self {
        let mut axis = axis.into();
        axis.sort_unstable_by(|a, b| b.cmp(a));
        Self {
            inp,
            out,
            axis,
            keep_dims,
        }
    }
}

impl<D: Floating> Op<D> for Sum {
    fn name(&self) -> &'static str {
        "sum"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let t_in = ctx.checked_get(&self.inp);

        let result = if self.axis.is_empty() {
            // If no axes are specified, sum all elements to a scalar.
            let sum_val = t_in.sum();
            ndarray::arr0(sum_val).into_dyn()
        } else {
            // else sum along the specified axes.
            let mut t = t_in;
            for axis in &self.axis {
                let a = Axis(*axis);
                t = if self.keep_dims {
                    t.sum_axis(a).insert_axis(a)
                } else {
                    t.sum_axis(a)
                }
            }
            t
        };

        ctx.insert(self.out, result);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // d/dx sum(x, axis) = broadcast_like(og, like=x)
        let grad_y = *out_grads.first()?;
        let reshaped_grad_id = g.fresh();

        g.push(Box::new(ReshapeForBroadcast::new(
            grad_y,
            reshaped_grad_id,
            self.axis.clone(),
            self.keep_dims,
        )));

        let broadcast_out_id = g.fresh();
        g.push(Box::new(BroadcastLike::new(
            reshaped_grad_id,
            self.inp,
            broadcast_out_id,
        )));

        Some(vec![broadcast_out_id])
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}

impl Tracer {
    pub fn sum(&self, _axis: impl Into<Vec<usize>>, _keep_dims: bool) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    pub fn sum(&mut self, a: Tracer, axis: impl Into<Vec<usize>>, keep_dims: bool) -> Tracer {
        let out = self.g.fresh();
        self.emit(Sum::new(a.id(), out, axis, keep_dims), out)
    }
}

// Reduce (sum) runtime `inp` down to the runtime shape of `like`.
#[derive(Debug, Clone)]
pub struct ReduceToLike {
    inp: Id,
    like: Id,
    out: Id,
}

impl ReduceToLike {
    pub fn new(inp: Id, like: Id, out: Id) -> Self {
        Self { inp, like, out }
    }
}

impl<D: Floating> Op<D> for ReduceToLike {
    fn name(&self) -> &'static str {
        "reduce_to_like"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        use ndarray::Axis;

        let inp = ctx.checked_get(&self.inp);
        let like = ctx.checked_get(&self.like);
        let a_shape = inp.shape();
        let b_shape = like.shape();

        if a_shape == b_shape {
            ctx.insert(self.out, inp.clone());
            return;
        }

        assert!(
            a_shape.len() >= b_shape.len(),
            "reduce_to_like: rank(inp) < rank(like). inp: {:?}, like: {:?}",
            a_shape,
            b_shape
        );

        use itertools::EitherOrBoth::{Both, Left};
        let t = a_shape
            .iter()
            .enumerate()
            .rev()
            .zip_longest(b_shape.iter().rev())
            .fold(t, |acc, tuple| {
                match tuple {
                    Left((axis,_)) => acc.sum_axis(Axis(axis)),
                    Both((_, &a), &b) if a == b => acc, // same dim, do nothing.
                    Both((axis, _), &1) => acc.sum_axis(Axis(axis)).insert_axis(Axis(axis)),
                    _ => panic!(
                        "reduce_to_like: cannot reduce inp -> like:  inp: {a_shape:?}, like: {b_shape:?}"
                    ),
                }
            });

        assert_eq!(
            t.shape(),
            like.shape(),
            "reduce_to_like: shapes mismatch after reduction: inp {:?} like {:?} got {:?}",
            a_shape,
            b_shape,
            t.shape()
        );

        ctx.insert(self.out, t);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // grad wrt inp = broadcast_like(og, like=inp)
        let og = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(BroadcastLike::new(og, self.inp, out)));
        Some(vec![out])
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp, self.like]
    }
    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}

#[derive(Debug, Clone)]
pub struct ReshapeForBroadcast {
    inp_grad: Id,
    out: Id,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl ReshapeForBroadcast {
    pub fn new(inp_grad: Id, out: Id, axis: impl Into<Vec<usize>>, keep_dims: bool) -> Self {
        Self {
            inp_grad,
            out,
            axis: axis.into(),
            keep_dims,
        }
    }
}

impl<D: Floating> Op<D> for ReshapeForBroadcast {
    fn name(&self) -> &'static str {
        "reshape_for_broadcast"
    }
    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp_grad]
    }
    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
    fn vjp(&self, _: &mut Graph<D>, _: &[Id]) -> Option<Vec<Id>> {
        None
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let inp_grad_tensor = ctx.checked_get(&self.inp_grad);

        // If keep_dims was true, or if it was a full reduction to a scalar,
        // the shape is already correct for broadcasting. No op needed.
        if self.keep_dims || self.axis.is_empty() {
            ctx.insert(self.out, inp_grad_tensor.clone());
            return;
        }

        let mut intermediate_shape = inp_grad_tensor.shape().to_vec();
        let mut sorted_axes = self.axis.clone();
        sorted_axes.sort_unstable(); // Sort to insert into the correct positions

        for &axis in &sorted_axes {
            intermediate_shape.insert(axis, 1);
        }

        let reshaped_tensor = inp_grad_tensor
            .to_shape(intermediate_shape)
            .unwrap()
            .to_owned()
            .into_dyn();
        ctx.insert(self.out, reshaped_tensor);
    }
}
