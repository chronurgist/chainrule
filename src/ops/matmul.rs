use crate::{
    Graph, Tracer,
    context::Context,
    identity::Id,
    ops::{Op, transpose::TransposeDefault},
};
use ndarray::{
    Array, ArrayD, ArrayView1, Ix1, Ix2, IxDyn,
    linalg::{general_mat_mul, general_mat_vec_mul},
};

use crate::{Floating, tracing::TensorData};

fn batched_matmul<D: Floating + 'static>(a: &ArrayD<D>, b: &ArrayD<D>) -> ArrayD<D> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    assert!(
        shape_a.len() >= 2 && shape_b.len() >= 2,
        "inputs for batched matrix mul should have rank > 2"
    );

    let (m, k1) = (shape_a[shape_a.len() - 2], shape_a[shape_a.len() - 1]);
    let (k2, n) = (shape_b[shape_b.len() - 2], shape_b[shape_b.len() - 1]);
    assert_eq!(
        k1, k2,
        "inner matrix dimensions should match for matrix mul: lhs contracted dim is {k1}, rhs is {k2}"
    );

    let batch_a = &shape_a[..shape_a.len() - 2];
    let batch_b = &shape_b[..shape_b.len() - 2];
    let batch_shape = super::broadcast_shapes(batch_a, batch_b)
        .expect("batch dimensions should be broadcast-compatible");

    let bc_shape_a: Vec<usize> = batch_shape.iter().copied().chain([m, k1]).collect();
    let bc_shape_b: Vec<usize> = batch_shape.iter().copied().chain([k2, n]).collect();

    let a_bc = a
        .broadcast(IxDyn(&bc_shape_a))
        .expect("broadcasting to a derived valid shape should be infallible ");
    let b_bc = b
        .broadcast(IxDyn(&bc_shape_b))
        .expect("broadcasting to a derived valid shape should be infallible ");

    let result_shape: Vec<usize> = batch_shape.iter().copied().chain([m, n]).collect();
    let mut result = ArrayD::zeros(IxDyn(&result_shape));

    let batch_elems: usize = batch_shape.iter().product();
    let a_reshaped = a_bc
        .to_shape((batch_elems, m, k1))
        .expect("reshape should succeed because the number of elements is preserved");
    let b_reshaped = b_bc
        .to_shape((batch_elems, k2, n))
        .expect("reshape should succeed because the number of elements is preserved");
    let binding = result.view_mut();
    let mut r_reshaped = binding
        .to_shape((batch_elems, m, n))
        .expect("reshape should succeed because the number of elements is preserved");

    ndarray::Zip::from(a_reshaped.outer_iter())
        .and(b_reshaped.outer_iter())
        .and(r_reshaped.outer_iter_mut())
        .for_each(|ai, bi, mut ri| {
            general_mat_mul(D::one(), &ai, &bi, D::zero(), &mut ri);
        });

    result
}

pub fn matmul<D: Floating + 'static>(a: &TensorData<D>, b: &TensorData<D>) -> TensorData<D> {
    match (a.ndim(), b.ndim()) {
        // scalar
        (0, _) | (_, 0) => a * b,

        // vector dot product
        (1, 1) => {
            assert_eq!(
                a.len(),
                b.len(),
                "vectors in dot-product should have same length"
            );
            let a1: ArrayView1<D> = a
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=1 tensor should be convertible to a 1D view");
            let b1: ArrayView1<D> = b
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=1 tensor should be convertible to a 1D view");
            TensorData::from_elem(vec![], a1.dot(&b1))
        }

        // vector (a or b) @ matrix (a, b) -> vector (1D)
        (1, 2) => {
            let n = a.len();
            assert_eq!(
                n,
                b.shape()[0],
                "vector length should match matrix's outer dimension for vec @ mat"
            ); // (n,) @ (n,m)
            let m = b.shape()[1];
            let a1 = a
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=1 tensor should be convertible to a 1D view");
            let b2 = b
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");

            let mut result = Array::zeros(m);
            // (1×n) × (n×m) → (m,)
            general_mat_vec_mul(D::one(), &b2.t(), &a1, D::zero(), &mut result);
            result.into_dyn()
        }

        // matrix (a, b) @ vector (a or b) -> vector (1D)
        (2, 1) => {
            let n = b.len();
            assert_eq!(
                n,
                a.shape()[1],
                "vector length should match matrix's inner dimension for mat @ vec"
            ); // (m,n) @ (n,)
            let m = a.shape()[0];
            let a2 = a
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");
            let b1 = b
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");

            let mut result = Array::zeros(m);
            general_mat_vec_mul(D::one(), &a2, &b1, D::zero(), &mut result);
            result.into_dyn()
        }

        // matrix (a,b) @ matrix (b, c) -> matrix (a, c)
        (2, 2) => {
            let (m, k1) = (a.shape()[0], a.shape()[1]);
            let (k2, n) = (b.shape()[0], b.shape()[1]);
            assert_eq!(
                k1, k2,
                "inner dimension for matrix mul should be equal but lhs({k1}) != rhs({k2})"
            );

            let a2 = a
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");
            let b2 = b
                .view()
                .into_dimensionality::<Ix2>()
                .expect("an ndim=2 tensor should be convertible to a 2D view");

            let mut result = Array::zeros((m, n));
            general_mat_mul(D::one(), &a2, &b2, D::zero(), &mut result);
            result.into_dyn()
        }

        // fallback to batched dims
        _ => batched_matmul(a, b),
    }
}

#[derive(Debug, Clone)]
pub struct MatMul {
    pub lhs: Id,
    pub rhs: Id,
    pub out: Id,
}

impl MatMul {
    pub fn new(lhs: Id, rhs: Id, out: Id) -> Self {
        Self { lhs, rhs, out }
    }
}

impl<D: Floating + 'static> Op<D> for MatMul {
    fn name(&self) -> &str {
        "matmul"
    }

    fn inputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.lhs, self.rhs]
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let lhs = ctx.checked_get(&self.lhs);
        let rhs = ctx.checked_get(&self.rhs);
        ctx.insert(self.out, matmul(lhs, rhs));
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let og = *out_grads.first()?;

        let rhs_t = {
            let out = g.fresh();
            let transpose = TransposeDefault::new(self.rhs, out);
            g.push(Box::new(transpose));
            out
        };

        let grad_lhs = {
            let out = g.fresh();
            let matmul = MatMul::new(og, rhs_t, out);
            g.push(Box::new(matmul));
            out
        };

        let lhs_t = {
            let out = g.fresh();
            let transpose = TransposeDefault::new(self.lhs, out);
            g.push(Box::new(transpose));
            out
        };

        let grad_rhs = {
            let out = g.fresh();
            let matmul = MatMul::new(lhs_t, og, out);
            g.push(Box::new(matmul));
            out
        };

        Some(vec![grad_lhs, grad_rhs])
    }
}

impl<D: Floating + 'static> crate::tracing::session::TraceSession<'_, D> {
    #[must_use]
    pub fn matmul(&mut self, a: Tracer, b: Tracer) -> Tracer {
        let out = self.g.fresh();
        self.emit(MatMul::new(a.id(), b.id(), out), out)
    }
}

impl Tracer {
    pub fn matmul(&self, _: Tracer) -> Tracer {
        panic!("dummy operation - only allowed inside #[trace] function")
    }
}

pub fn infer_matmul_shape(lhs: &[usize], rhs: &[usize]) -> Vec<usize> {
    match (lhs.len(), rhs.len()) {
        // scalar x anything → result shape is other
        (0, _) => rhs.to_vec(),
        (_, 0) => lhs.to_vec(),

        // vector dot product (n,) @ (n,) -> scalar
        (1, 1) => vec![],

        // (n,) @ (n,m) -> (m,)
        (1, 2) => vec![rhs[1]],

        // (m,n) @ (n,) -> (m,)
        (2, 1) => vec![lhs[0]],

        // (m,k) @ (k,n) -> (m,n)
        (2, 2) => vec![lhs[0], rhs[1]],

        // general batched case
        _ => {
            let batch_a = &lhs[..lhs.len() - 2];
            let batch_b = &rhs[..rhs.len() - 2];
            let batch_shape = super::broadcast_shapes(batch_a, batch_b)
                .expect("batch dims broadcastable for matmul");

            let m = lhs[lhs.len() - 2];
            let n = rhs[rhs.len() - 1];
            let mut result = batch_shape;
            result.push(m);
            result.push(n);
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use crate::prelude::*;

    #[test]
    fn test_matmul() {
        #[trace]
        fn f(x: Tensor, w: Tensor) -> Tensor {
            x.matmul(w)
        }

        let traced = trace_fn::<f32>(f);

        let x = arr2(&[[1., 2.], [3., 4.]]);
        let w = arr2(&[[5., 6.], [7., 8.]]);
        let x2 = arr2(&[[1., 2.], [3., 4.]]).into_dyn();
        let w2 = arr2(&[[5., 6.], [7., 8.]]).into_dyn();
        let (out,) = traced.eval()((&x2, &w2));
        let expected = x.dot(&w);
        assert_eq!(out, expected.into_dyn());
    }
}
