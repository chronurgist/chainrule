use crate::{Floating, context::Context, graph::Graph, identity::Id, ops::Op};

#[derive(Debug, Clone)]
pub struct Const<D: Floating> {
    pub value: D,
    pub out: Id,
}

impl<D: Floating> Const<D> {
    pub fn new(value: D, out: Id) -> Self {
        Self { value, out }
    }
    pub fn boxed(value: D, out: Id) -> Box<Self> {
        Box::new(Self::new(value, out))
    }
}

impl<D: Floating + 'static> Op<D> for Const<D> {
    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![]
    }
    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
    fn eval(&self, ctx: &mut Context<D>) {
        use ndarray::arr0;
        ctx.insert(self.out, arr0(self.value).into_dyn());
    }
    fn vjp(&self, _g: &mut Graph<D>, _out_grads: &[Id]) -> Option<Vec<Id>> {
        vec![].into()
    }

    fn name(&self) -> &'static str {
        "const"
    }
}
