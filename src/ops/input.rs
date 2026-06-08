use crate::{Floating, context::Context, graph::Graph, identity::Id, ops::Op};

#[derive(Debug, Clone)]
pub struct Input {
    pub out: Id,
}

impl Input {
    pub fn new(out: Id) -> Self {
        Self { out }
    }
}

impl<D: Floating + 'static> Op<D> for Input {
    fn name(&self) -> &'static str {
        "input"
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }

    fn eval(&self, _ctx: &mut Context<D>) {
        // no-op: input tensors are already loaded into Context by TraceableFn::eval
    }

    fn vjp(&self, _g: &mut Graph<D>, _out_grads: &[Id]) -> Option<Vec<Id>> {
        // no grads for inputs, this is just a load operation
        None
    }
}
