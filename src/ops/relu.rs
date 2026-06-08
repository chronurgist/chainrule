use crate::{
    Floating, Graph, Id, TraceSession, Tracer,
    context::Context,
    ops::{Mul, Op},
    simple_unary_op,
    tracing::TensorData,
};

simple_unary_op!(
    ReLU,
    disp: "relu",
    fwd: |x: &TensorData<D>| x.mapv(|a| if a > D::zero() { a } else { D::zero() }),
    vjp: |this: &ReLU, g: &mut Graph<D>, og: Id| {
        // grad = og * 1[x>0]
        let mask_out = g.fresh();
        g.push(Box::new(ReLUGradMask::new(this.inp, mask_out)));
        let prod = g.fresh();
        g.push(Box::new(Mul::new(og, mask_out, prod)));
        prod
    }
);

#[derive(Debug, Clone)]
pub struct ReLUGradMask {
    inp: Id,
    out: Id,
}
impl ReLUGradMask {
    pub fn new(inp: Id, out: Id) -> Self {
        Self { inp, out }
    }
}
impl<D: Floating + 'static> Op<D> for ReLUGradMask {
    fn name(&self) -> &str {
        "relu_mask"
    }
    fn eval(&self, ctx: &mut Context<D>) {
        let x = ctx.checked_get(&self.inp);
        let mask = x.mapv(|a| if a > D::zero() { D::one() } else { D::zero() });
        ctx.insert(self.out, mask);
    }
    fn vjp(&self, _g: &mut Graph<D>, _og: &[Id]) -> Option<Vec<Id>> {
        // d(1[x>0])/dx is 0 almost everywhere, so no backward pass
        None
    }
    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp]
    }
    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}

impl Tracer {
    pub fn relu(&self) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    pub fn relu(&mut self, a: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(ReLU::new(a.id(), out), out)
    }
}
