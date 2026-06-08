use crate::{Floating, Graph, Id, TraceSession, Tracer, context::Context, ops::Op};

#[derive(Debug, Clone)]
pub struct Reshape {
    inp: Id,
    out: Id,
    target_shape: Vec<usize>,
}

impl Reshape {
    pub fn new(inp: Id, out: Id, target_shape: impl Into<Vec<usize>>) -> Self {
        Self {
            inp,
            out,
            target_shape: target_shape.into(),
        }
    }
}

impl<D: Floating> Op<D> for Reshape {
    fn name(&self) -> &'static str {
        "reshape"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let t = ctx.checked_get(&self.inp);
        let reshaped = t
            .to_shape(&*self.target_shape)
            .expect("reshape should succeed as the number of elements is preserved");
        ctx.insert(self.out, reshaped.to_owned());
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // d/dx reshape(x -> target) = reshape_like(og, like=x)
        let grad_y = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(ReshapeLike::new(grad_y, out, self.inp)));
        Some(vec![out])
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}

impl<D: Floating + 'static> TraceSession<'_, D> {
    #[must_use]
    pub fn reshape(&mut self, t: Tracer, shape: impl Into<Vec<usize>>) -> Tracer {
        let out = self.g.fresh();
        self.emit(Reshape::new(t.id(), out, shape), out)
    }
}

impl Tracer {
    pub fn reshape(&self, _: impl Into<Vec<usize>>) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

// Reshape to the runtime shape of `like`.
#[derive(Debug, Clone)]
pub struct ReshapeLike {
    inp: Id,
    out: Id,
    like: Id,
}

impl ReshapeLike {
    pub fn new(inp: Id, out: Id, like: Id) -> Self {
        Self { inp, out, like }
    }
}

impl<D: Floating> Op<D> for ReshapeLike {
    fn name(&self) -> &'static str {
        "reshape_like"
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let x = ctx.checked_get(&self.inp);
        let like = ctx.checked_get(&self.like);
        let target = like.shape().to_vec();
        let y = x
            .to_shape(target)
            .expect("reshape_like: element count mismatch")
            .to_owned();
        ctx.insert(self.out, y);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        // its own inverse: reshape_like(og, like=inp)
        let og = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(ReshapeLike::new(og, out, self.inp)));
        Some(vec![out])
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.inp, self.like]
    }
    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }
}
