use crate::{
    Graph, Id,
    ops::{Const, Mul, Neg, sum::ReduceToLike},
    primitive_binary_op,
    tracing::TensorData,
};

primitive_binary_op!(
    Div,
    disp: "div",
    fwd: |x: &TensorData<D>, y: &TensorData<D>| x / y,
    vjp: |this: &Div, g: &mut Graph<D>, og: Id| {
        // d/dx (x/y) = og * (1/y)
        let one_id = {
            let id = g.fresh();
            g.push(Box::new(Const::new(D::one(), id)));
            id
        };
        let inv_rhs = {
            let out = g.fresh();
            g.push(Box::new(Div::new(one_id, this.rhs, out)));
            out
        };
        let grad_lhs = {
            let prod = g.fresh();
            g.push(Box::new(Mul::new(og, inv_rhs, prod)));
            let out = g.fresh();
            g.push(Box::new(ReduceToLike::new(prod, this.lhs, out)));
            out
        };
        // d/dy (x/y) = og * (-x / y^2)
        let y2 = {
            let out = g.fresh();
            g.push(Box::new(Mul::new(this.rhs, this.rhs, out)));
            out
        };
        let neg_x = {
            let out = g.fresh();
            g.push(Box::new(Neg::new(this.lhs, out)));
            out
        };
        let grad_rhs = {
            let prod = g.fresh();
            g.push(Box::new(Div::new(neg_x, y2, prod)));
            let out = g.fresh();
            g.push(Box::new(Mul::new(og, prod, out)));
            let reduced = g.fresh();
            g.push(Box::new(ReduceToLike::new(out, this.rhs, reduced)));
            reduced
        };
        vec![grad_lhs, grad_rhs]
    }
);
