use ndarray::Axis;

use crate::{
    Floating, Graph, Id, TraceSession, Tracer,
    context::Context,
    ops::{Op, broadcast::BroadcastLike, div::Div, mul::Mul, sum::Sum},
};

#[derive(Debug, Clone)]
pub struct Max {
    inp: Id,
    out: Id,
    axis: Vec<usize>,
    keep_dims: bool,
}

impl Max {
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

impl<D: Floating + 'static> Op<D> for Max {
    fn name(&self) -> &'static str {
        "max"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let mut t = ctx.checked_get(&self.inp).clone();
        for ax in &self.axis {
            let a = Axis(*ax);
            let reduced = t.fold_axis(
                a,
                D::neg_infinity(),
                |acc, x| if acc > x { *acc } else { *x },
            );

            t = if self.keep_dims {
                reduced.insert_axis(a)
            } else {
                reduced
            };
        }
        ctx.insert(self.out, t);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // grad wrt x:
        // - Broadcast og to x's shape
        // - Broadcast y (max result) back to x's shape
        // - mask = 1[x == y_broadcast]
        // - count = sum(mask, axis)
        // - grad = (og_broadcast * mask) / broadcast_like(count, like=x)
        let og = *out_grads.first()?;

        let og_bc = {
            let out = g.fresh();
            g.push(Box::new(BroadcastLike::new(og, self.inp, out)));
            out
        };

        let y_bc = {
            let out = g.fresh();
            g.push(Box::new(BroadcastLike::new(self.out, self.inp, out)));
            out
        };

        let mask = {
            let out = g.fresh();
            g.push(Box::new(MaxGradMask::new(self.inp, y_bc, out)));
            out
        };

        let count_y_shape = {
            let out = g.fresh();
            g.push(Box::new(Sum::new(
                mask,
                out,
                self.axis.clone(),
                self.keep_dims,
            )));
            out
        };

        let count_bc = {
            let out = g.fresh();
            g.push(Box::new(BroadcastLike::new(count_y_shape, self.inp, out)));
            out
        };

        let numer = {
            let out = g.fresh();
            g.push(Box::new(Mul::new(og_bc, mask, out)));
            out
        };

        let grad_x = {
            let out = g.fresh();
            g.push(Box::new(Div::new(numer, count_bc, out)));
            out
        };

        Some(vec![grad_x])
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}

// Backward helper: produce a mask 1.0 where x == y, else 0.0
#[derive(Debug, Clone)]
pub struct MaxGradMask {
    x: Id,
    y: Id, // same shape as x
    out: Id,
}

impl MaxGradMask {
    pub fn new(x: Id, y: Id, out: Id) -> Self {
        Self { x, y, out }
    }
}

impl<D: Floating + 'static> Op<D> for MaxGradMask {
    fn name(&self) -> &'static str {
        "max_mask"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let x = ctx.checked_get(&self.x);
        let y = ctx.checked_get(&self.y);
        assert_eq!(
            x.shape(),
            y.shape(),
            "max grad mask: x and y must have the same shape"
        );

        let mask = ndarray::Zip::from(x.view())
            .and(y.view())
            .map_collect(|&a, &b| if a == b { D::one() } else { D::zero() });

        ctx.insert(self.out, mask);
    }

    fn vjp(&self, _g: &mut Graph<D>, _out_grads: &[Id]) -> Option<Vec<Id>> {
        // Derivative of the indicator (almost everywhere) is zero; no backward pass
        None
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.x, self.y]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}

impl Tracer {
    pub fn max(&self, _axis: impl Into<Vec<usize>>, _keep_dims: bool) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    pub fn max(&mut self, a: Tracer, axis: impl Into<Vec<usize>>, keep_dims: bool) -> Tracer {
        let out = self.g.fresh();
        self.emit(Max::new(a.id(), out, axis, keep_dims), out)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_max_forward() {
        use crate::prelude::*;
        use ndarray::arr2;

        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.max(vec![1], false)
        }

        let traced = trace_fn::<f32>(f);
        let x = arr2(&[[1., 3., 2.], [4., 0., 4.]]).into_dyn();
        let (out,) = traced.eval()(&x);
        let expected = x
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()
            .map_axis(ndarray::Axis(1), |lane| {
                lane.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            })
            .into_dyn();
        assert_eq!(out, expected);
    }
}
