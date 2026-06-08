use ndarray::Axis;

use crate::{
    Floating, Graph, Id, TraceSession, Tracer,
    context::Context,
    ops::{
        Op,
        broadcast::BroadcastLike,
        constant::Const,
        div::Div,
        sum::{ReshapeForBroadcast, Sum},
    },
};

#[derive(Debug, Clone)]
pub struct Mean {
    inp: Id,
    out: Id,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl Mean {
    pub fn new(inp: Id, out: Id, axis: impl Into<Vec<usize>>, keep_dims: bool) -> Self {
        let mut axis = axis.into();
        // Reduce higher axes first to keep indexing valid as dims shrink
        axis.sort_unstable_by(|a, b| b.cmp(a));
        Self {
            inp,
            out,
            axis,
            keep_dims,
        }
    }
}

impl<D: Floating + 'static> Op<D> for Mean {
    fn name(&self) -> &'static str {
        "mean"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let x = ctx.checked_get(&self.inp);
        // sum along axes
        if self.axis.is_empty() {
            // sum to scalar.
            let n = D::from_f64(x.len() as f64);
            if n == D::zero() {
                // div by zero -> 0
                ctx.insert(self.out, ndarray::arr0(D::zero()).into_dyn());
                return;
            }
            let sum = x.sum();
            let result = ndarray::arr0(sum / n).into_dyn();
            ctx.insert(self.out, result);
            return;
        }

        for ax in &self.axis {
            let a = Axis(*ax);
            t = if self.keep_dims {
                t.sum_axis(a).insert_axis(a)
            } else {
                t.sum_axis(a)
            };
        }

        // divide by the count of reduced elements
        let shape = x.shape();
        let mut denom = D::one();
        for &ax in &self.axis {
            denom = denom * D::from_f64(shape[ax] as f64);
        }
        // If no axes provided, denom=1 and t==x; that's fine.
        let t = t.mapv(|v| v / denom);
        ctx.insert(self.out, t);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // grad wrt x:
        // - Create ones_like(x) by broadcasting scalar 1 to x's shape
        // - counts_y = sum(ones_like(x), axis)  // same shape as output y
        // - counts_bc = broadcast_like(counts_y, like=x)
        // - og_bc = broadcast_like(og, like=x)
        // - grad = og_bc / counts_bc
        let og = *out_grads.first()?;

        // Step 1: Calculate the scaling factor (1 / N)
        let one = {
            let id = g.fresh();
            g.push(Box::new(Const::new(D::one(), id)));
            id
        };
        let ones_like_x = {
            let out = g.fresh();
            g.push(Box::new(BroadcastLike::new(one, self.inp, out)));
            out
        };
        let counts_y = {
            let out = g.fresh();
            g.push(Box::new(Sum::new(
                ones_like_x,
                out,
                self.axis.clone(),
                self.keep_dims,
            )));
            out
        };

        let scaled_og = {
            let out = g.fresh();
            g.push(Box::new(Div::new(og, counts_y, out)));
            out
        };

        // reshape the scaled gradient to make it broadcast-compatible
        let reshaped_grad_id = g.fresh();
        g.push(Box::new(ReshapeForBroadcast::new(
            scaled_og,
            reshaped_grad_id,
            self.axis.clone(),
            self.keep_dims,
        )));

        let grad_x = g.fresh();
        g.push(Box::new(BroadcastLike::new(
            reshaped_grad_id,
            self.inp,
            grad_x,
        )));

        Some(vec![grad_x])
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}

impl Tracer {
    pub fn mean(&self, _axis: impl Into<Vec<usize>>, _keep_dims: bool) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    pub fn mean(&mut self, a: Tracer, axis: impl Into<Vec<usize>>, keep_dims: bool) -> Tracer {
        let out = self.g.fresh();
        self.emit(Mean::new(a.id(), out, axis, keep_dims), out)
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn test_mean_forward() {
        use crate::prelude::*;
        use ndarray::arr2;

        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.mean(vec![1], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr2(&[[1., 3., 2.], [4., 0., 4.]]).into_dyn();
        let (out,) = traced.eval()(&x);
        let expected = x
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .into_dyn();
        assert_eq!(out, expected);
    }
}
