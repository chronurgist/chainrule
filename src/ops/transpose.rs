use crate::{Tracer, context::Context, graph::Graph, identity::Id};

use crate::{Floating, ops::Op};

#[derive(Debug, Clone)]
pub struct TransposeDefault {
    pub inp: Id,
    pub out: Id,
}

impl TransposeDefault {
    pub fn new(inp: Id, out: Id) -> Self {
        Self { inp, out }
    }
}

impl<D: Floating + 'static> Op<D> for TransposeDefault {
    fn name(&self) -> &str {
        "transpose_default"
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let t = ctx.checked_get(&self.inp);
        let shape = t.shape();
        let rank = shape.len();
        if rank > 1 {
            let mut axes: Vec<usize> = (0..rank).collect();
            axes.swap(rank - 1, rank - 2);
            ctx.insert(self.out, t.view().permuted_axes(axes).to_owned());
        } else {
            ctx.insert(self.out, t.clone());
        }
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let og = *out_grads.first()?;
        let out = g.fresh();
        let transpose = Box::new(TransposeDefault::new(og, out));
        g.push(transpose);
        Some(vec![out])
    }
}

#[derive(Debug, Clone)]
pub struct Transpose {
    pub inp: Id,
    pub out: Id,
    pub a1: usize,
    pub a2: usize,
}

impl Transpose {
    pub fn new(inp: Id, out: Id, a1: usize, a2: usize) -> Self {
        Self { inp, out, a1, a2 }
    }
}

impl<D: Floating + 'static> Op<D> for Transpose {
    fn name(&self) -> &str {
        "transpose"
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let t = ctx.checked_get(&self.inp);
        let rank = t.ndim();
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(self.a1, self.a2);
        ctx.insert(self.out, t.view().permuted_axes(axes).to_owned());
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let og = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(Transpose::new(og, out, self.a1, self.a2)));
        Some(vec![out])
    }
}

impl<D: Floating + 'static> crate::tracing::session::TraceSession<'_, D> {
    #[must_use]
    pub fn t(&mut self, a: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(TransposeDefault::new(a.id(), out), out)
    }

    #[must_use]
    pub fn transpose(&mut self, a: Tracer, a1: usize, a2: usize) -> Tracer {
        let out = self.g.fresh();
        self.emit(Transpose::new(a.id(), out, a1, a2), out)
    }
}

impl Tracer {
    pub fn transpose(&self, _: usize, _: usize) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
    pub fn t(&self) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}
#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use crate::prelude::*;

    #[test]
    fn test_transpose_default() {
        #[trace]
        fn f(x: Tensor) -> Tensor {
            x.t()
        }

        let traced = trace_fn::<f32>(f);

        let x = arr2(&[[1., 2., 3.], [4., 5., 6.]]).into_dyn();
        let (out,) = traced.eval()(&x);
        let expected = x.t().into_owned().into_dyn();
        assert_eq!(out, expected);
    }
}
