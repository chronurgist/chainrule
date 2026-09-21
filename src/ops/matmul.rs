use crate::{
    Graph, Tracer,
    context::Context,
    identity::Id,
    ops::{Op, reshape::ReshapeLike, sum::ReduceToLike, transpose::TransposeDefault},
};
use ndarray::{
    Array, ArrayD, ArrayViewD, Ix1, Ix2, IxDyn,
    linalg::{general_mat_mul as gemm_impl, general_mat_vec_mul as gemv_impl},
};

use crate::{Floating, tracing::TensorData};
use smallvec::{SmallVec, smallvec};

type Shape = SmallVec<[usize; 4]>;

fn batched_matmul<D: Floating + 'static>(a: ArrayViewD<'_, D>, b: ArrayViewD<'_, D>) -> ArrayD<D> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    assert!(
        shape_a.len() >= 2 && shape_b.len() >= 2,
        "inputs for batched matrix mul should have rank >= 2"
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

    let bc_shape_a: Shape = batch_shape.iter().copied().chain([m, k1]).collect();
    let bc_shape_b: Shape = batch_shape.iter().copied().chain([k2, n]).collect();

    let a_bc = a
        .broadcast(IxDyn(&bc_shape_a))
        .expect("broadcasting to a derived valid shape should be infallible ")
        .to_owned();
    let b_bc = b
        .broadcast(IxDyn(&bc_shape_b))
        .expect("broadcasting to a derived valid shape should be infallible ")
        .to_owned();

    let result_shape: Shape = batch_shape.iter().copied().chain([m, n]).collect();
    let mut result = ArrayD::zeros(IxDyn(&result_shape));

    let batch_elems: usize = batch_shape.iter().product();
    let a_reshaped = a_bc
        .to_shape((batch_elems, m, k1))
        .expect("reshape should succeed because the number of elements is preserved");
    let b_reshaped = b_bc
        .to_shape((batch_elems, k2, n))
        .expect("reshape should succeed because the number of elements is preserved");
    let mut r_reshaped = result
        .view_mut()
        .into_shape_with_order((batch_elems, m, n))
        .expect("result has standard layout");

    ndarray::Zip::from(a_reshaped.outer_iter())
        .and(b_reshaped.outer_iter())
        .and(r_reshaped.outer_iter_mut())
        .for_each(|ai, bi, mut ri| {
            gemm_impl(D::one(), &ai, &bi, D::zero(), &mut ri);
        });

    result
}

pub fn matmul<D: Floating + 'static>(a: &TensorData<D>, b: &TensorData<D>) -> TensorData<D> {
    assert!(
        a.ndim() > 0 && b.ndim() > 0,
        "matmul inputs must have rank >= 1; use * for scalars"
    );
    match (a.ndim(), b.ndim()) {
        // vector dot product
        (1, 1) => {
            assert_eq!(
                a.len(),
                b.len(),
                "vectors in dot-product should have same length"
            );
            let a1 = a
                .view()
                .into_dimensionality::<Ix1>()
                .expect("an ndim=1 tensor should be convertible to a 1D view");
            let b1 = b
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
            gemv_impl(D::one(), &b2.t(), &a1, D::zero(), &mut result);
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
            gemv_impl(D::one(), &a2, &b1, D::zero(), &mut result);
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
            gemm_impl(D::one(), &a2, &b2, D::zero(), &mut result);
            result.into_dyn()
        }

        // Promote vectors before batch multiplication, then remove their output axes.
        _ => {
            let shapes = MatMulShapes::new(a.shape(), b.shape());
            let a = a
                .to_shape(IxDyn(&shapes.lhs))
                .expect("vector promotion preserves elements");
            let b = b
                .to_shape(IxDyn(&shapes.rhs))
                .expect("vector promotion preserves elements");
            batched_matmul(a.view(), b.view())
                .into_shape_with_order(IxDyn(&shapes.output))
                .expect("removing vector axes preserves elements")
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MatMulShape {
    Left,
    Right,
    Output { lhs: Id, rhs: Id },
}

/// Promotes a vector to a row (left) or column (right), leaving higher ranks intact.
fn promote_shape(shape: &[usize], left: bool) -> Shape {
    assert!(
        !shape.is_empty(),
        "matmul inputs must have rank >= 1; use * for scalars"
    );
    if shape.len() == 1 {
        if left {
            smallvec![1, shape[0]]
        } else {
            smallvec![shape[0], 1]
        }
    } else {
        Shape::from_slice(shape)
    }
}

struct MatMulShapes {
    lhs: Shape,
    rhs: Shape,
    matrix_output: Shape,
    output: Shape,
}

impl MatMulShapes {
    /// Validates a matmul and derives both normalized and public output shapes.
    fn new(lhs: &[usize], rhs: &[usize]) -> Self {
        let a = promote_shape(lhs, true);
        let b = promote_shape(rhs, false);
        assert_eq!(
            a[a.len() - 1],
            b[b.len() - 2],
            "matmul contracted dimensions must match"
        );
        let batch = Shape::from_vec(
            super::broadcast_shapes(&a[..a.len() - 2], &b[..b.len() - 2])
                .expect("matmul batch dimensions must be broadcast-compatible"),
        );
        let mut matrix_output = batch.clone();
        matrix_output.extend([a[a.len() - 2], b[b.len() - 1]]);
        let mut output = batch;
        if lhs.len() > 1 {
            output.push(a[a.len() - 2]);
        }
        if rhs.len() > 1 {
            output.push(b[b.len() - 1]);
        }
        Self {
            lhs: a,
            rhs: b,
            matrix_output,
            output,
        }
    }
}

#[derive(Debug, Clone)]
struct MatMulReshape {
    inp: Id,
    out: Id,
    shape: MatMulShape,
}

impl<D: Floating + 'static> Op<D> for MatMulReshape {
    fn name(&self) -> &str {
        "matmul_reshape"
    }

    fn inputs(&self) -> crate::ops::IdList {
        let mut inputs = smallvec::smallvec![self.inp];
        if let MatMulShape::Output { lhs, rhs } = self.shape {
            inputs.extend([lhs, rhs]);
        }
        inputs
    }

    fn outputs(&self) -> crate::ops::IdList {
        smallvec::smallvec![self.out]
    }

    fn eval(&self, ctx: &mut Context<D>) {
        let inp = ctx.checked_get(&self.inp);
        let shape = match self.shape {
            MatMulShape::Left => promote_shape(inp.shape(), true),
            MatMulShape::Right => promote_shape(inp.shape(), false),
            MatMulShape::Output { lhs, rhs } => {
                MatMulShapes::new(ctx.checked_get(&lhs).shape(), ctx.checked_get(&rhs).shape())
                    .matrix_output
            }
        };
        let value = inp
            .to_shape(IxDyn(&shape))
            .expect("matmul normalization preserves elements")
            .into_owned();
        ctx.insert(self.out, value);
    }

    fn vjp(&self, g: &mut Graph<D>, out_grads: &[Id]) -> Option<Vec<Id>> {
        let og = *out_grads.first()?;
        let out = g.fresh();
        g.push(Box::new(ReshapeLike::new(og, out, self.inp)));
        Some(vec![out])
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
        let a = g.fresh();
        let b = g.fresh();
        let grad = g.fresh();
        g.push(Box::new(MatMulReshape {
            inp: self.lhs,
            out: a,
            shape: MatMulShape::Left,
        }));
        g.push(Box::new(MatMulReshape {
            inp: self.rhs,
            out: b,
            shape: MatMulShape::Right,
        }));
        g.push(Box::new(MatMulReshape {
            inp: og,
            out: grad,
            shape: MatMulShape::Output {
                lhs: self.lhs,
                rhs: self.rhs,
            },
        }));

        let at = g.fresh();
        let bt = g.fresh();
        g.push(Box::new(TransposeDefault::new(a, at)));
        g.push(Box::new(TransposeDefault::new(b, bt)));
        let ga = g.fresh();
        let gb = g.fresh();
        g.push(Box::new(MatMul::new(grad, bt, ga)));
        g.push(Box::new(MatMul::new(at, grad, gb)));

        let mut grads = Vec::with_capacity(2);
        for (raw, normalized, original) in [(ga, a, self.lhs), (gb, b, self.rhs)] {
            let reduced = g.fresh();
            let out = g.fresh();
            g.push(Box::new(ReduceToLike::new(raw, normalized, reduced)));
            g.push(Box::new(ReshapeLike::new(reduced, out, original)));
            grads.push(out);
        }
        Some(grads)
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
    MatMulShapes::new(lhs, rhs).output.into_vec()
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn, arr1, arr2};

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

    #[test]
    fn test_matmul_vjp_rank_cases() {
        #[trace]
        fn f(a: Tensor, b: Tensor) -> Tensor {
            a.matmul(b)
        }

        let grad = trace_fn::<f32>(f).grad();

        let a = arr1(&[2., 3.]).into_dyn();
        let b = arr1(&[5., 7.]).into_dyn();
        let (ga, gb) = grad.eval()((&a, &b));
        assert_eq!(ga, arr1(&[5., 7.]).into_dyn());
        assert_eq!(gb, arr1(&[2., 3.]).into_dyn());

        let a = arr1(&[2., 3.]).into_dyn();
        let b = arr2(&[[5., 6., 7.], [8., 9., 10.]]).into_dyn();
        let (ga, gb) = grad.eval()((&a, &b));
        assert_eq!(ga, arr1(&[18., 27.]).into_dyn());
        assert_eq!(
            gb,
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2., 2., 2., 3., 3., 3.]).unwrap()
        );

        let a = arr2(&[[5., 6.], [7., 8.]]).into_dyn();
        let b = arr1(&[2., 3.]).into_dyn();
        let (ga, gb) = grad.eval()((&a, &b));
        assert_eq!(
            ga,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2., 3., 2., 3.]).unwrap()
        );
        assert_eq!(gb, arr1(&[12., 14.]).into_dyn());

        let a = arr2(&[[1., 2.], [3., 4.]]).into_dyn();
        let b = arr2(&[[5., 6.], [7., 8.]]).into_dyn();
        let (ga, gb) = grad.eval()((&a, &b));
        assert_eq!(
            ga,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![11., 15., 11., 15.]).unwrap()
        );
        assert_eq!(
            gb,
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![4., 4., 6., 6.]).unwrap()
        );
    }

    #[test]
    fn test_matmul_vjp_broadcasted_batches() {
        #[trace]
        fn f(a: Tensor, b: Tensor) -> Tensor {
            a.matmul(b)
        }
        let grad = trace_fn::<f32>(f).grad();
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2, 3]), (1..=12).map(|x| x as f32).collect())
            .unwrap();
        let b = ArrayD::from_shape_vec(IxDyn(&[1, 4, 3, 2]), (1..=24).map(|x| x as f32).collect())
            .unwrap();
        let (ga, gb) = grad.eval()((&a, &b));

        // The sum output seed is one.  Check both the non-broadcast and
        // broadcast axes, as well as the matrix contraction axes.
        let mut expected_a = ArrayD::zeros(IxDyn(&[2, 1, 2, 3]));
        let mut expected_b = ArrayD::zeros(IxDyn(&[1, 4, 3, 2]));
        for batch_a in 0..2 {
            for i in 0..2 {
                for r in 0..3 {
                    expected_a[[batch_a, 0, i, r]] = (0..4)
                        .map(|j| b[[0, j, r, 0]] + b[[0, j, r, 1]])
                        .sum::<f32>();
                }
            }
        }
        for j in 0..4 {
            for r in 0..3 {
                for batch_a in 0..2 {
                    for i in 0..2 {
                        expected_b[[0, j, r, 0]] += a[[batch_a, 0, i, r]];
                        expected_b[[0, j, r, 1]] += a[[batch_a, 0, i, r]];
                    }
                }
            }
        }
        assert_eq!(ga, expected_a);
        assert_eq!(gb, expected_b);
    }
}
