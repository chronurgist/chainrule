use chainrule::{
    ops::matmul::{infer_matmul_shape, matmul},
    prelude::*,
};
use ndarray::{ArrayD, IxDyn};

fn values(shape: &[usize], offset: f64) -> ArrayD<f64> {
    ArrayD::from_shape_fn(IxDyn(shape), |index| {
        use ndarray::Dimension;
        offset
            + index
                .slice()
                .iter()
                .enumerate()
                .map(|(axis, &i)| (axis + 1) as f64 * i as f64 * 0.13)
                .sum::<f64>()
    })
}

fn close(actual: &ArrayD<f64>, expected: &ArrayD<f64>) {
    assert_eq!(actual.shape(), expected.shape());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() < 1e-5 * (1.0 + e.abs()),
            "actual {a}, expected {e}"
        );
    }
}

/// Central differences keep the gradient checks independent of the symbolic VJP.
fn numerical_gradient(x: &ArrayD<f64>, f: impl Fn(&ArrayD<f64>) -> f64) -> ArrayD<f64> {
    let eps = 1e-5;
    let mut result = ArrayD::zeros(x.raw_dim());
    for (index, entry) in result.indexed_iter_mut() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[index.clone()] += eps;
        minus[index] -= eps;
        *entry = (f(&plus) - f(&minus)) / (2.0 * eps);
    }
    result
}

fn rank_cases() -> Vec<(Vec<usize>, Vec<usize>)> {
    vec![
        (vec![3], vec![3]),
        (vec![3], vec![3, 2]),
        (vec![2, 3], vec![3]),
        (vec![2, 3], vec![3, 4]),
        (vec![3], vec![2, 3, 4]),
        (vec![2, 4, 3], vec![3]),
        (vec![2, 3], vec![2, 3, 4]),
        (vec![2, 4, 3], vec![3, 2]),
        (vec![2, 1, 2, 3], vec![1, 3, 3, 2]),
        (vec![1, 2, 3], vec![2, 3, 4]),
    ]
}

#[test]
fn gradients_with_nonuniform_seed_across_runtime_ranks() {
    #[trace]
    fn weighted(a: Tensor, b: Tensor, weight: Tensor) -> Tensor {
        (a.matmul(b) * weight).sum(vec![], false)
    }
    let f = trace_fn::<f64>(weighted);
    let grad = f.grad();
    assert!(grad.graph.nodes.iter().all(|op| op.name() != "matmul_grad"));
    for (sa, sb) in rank_cases() {
        let a = values(&sa, 0.2);
        let b = values(&sb, -0.3);
        let shape = infer_matmul_shape(&sa, &sb);
        assert_eq!(matmul(&a, &b).shape(), shape);
        let weight = values(&shape, 0.7);
        let (ga, gb, _): (ArrayD<f64>, ArrayD<f64>, ArrayD<f64>) = grad.eval()((&a, &b, &weight));
        close(
            &ga,
            &numerical_gradient(&a, |x| (matmul(x, &b) * &weight).sum()),
        );
        close(
            &gb,
            &numerical_gradient(&b, |x| (matmul(&a, x) * &weight).sum()),
        );
    }
}

#[test]
fn second_derivatives_through_vectors_and_broadcasts() {
    #[trace]
    fn squared(a: Tensor, b: Tensor) -> Tensor {
        let y = a.matmul(b);
        (y * y).sum(vec![], false)
    }
    let grad = trace_fn::<f64>(squared).grad();
    // Differentiate each input gradient's sum separately: their shapes need not match.
    for selected in 0..2 {
        let mut first = grad.clone();
        first.outputs = vec![grad.outputs[selected]];
        let second = first.grad();
        for (sa, sb) in rank_cases() {
            let a = values(&sa, 0.2);
            let b = values(&sb, -0.3);
            let (ha, hb): (ArrayD<f64>, ArrayD<f64>) = second.eval()((&a, &b));
            close(
                &ha,
                &numerical_gradient(&a, |x| {
                    let (g,): (ArrayD<f64>,) = first.eval()((x, &b));
                    g.sum()
                }),
            );
            close(
                &hb,
                &numerical_gradient(&b, |x| {
                    let (g,): (ArrayD<f64>,) = first.eval()((&a, x));
                    g.sum()
                }),
            );
        }
    }
}

#[test]
fn vector_batch_forward_values() {
    let a = values(&[3], 0.2);
    let b = values(&[2, 3, 4], -0.3);
    let result = matmul(&a, &b);
    for batch in 0..2 {
        for j in 0..4 {
            let expected: f64 = (0..3).map(|k| a[[k]] * b[[batch, k, j]]).sum();
            assert!((result[[batch, j]] - expected).abs() < 1e-12);
        }
    }
    let a = values(&[2, 4, 3], 0.2);
    let b = values(&[3], -0.3);
    let result = matmul(&a, &b);
    for batch in 0..2 {
        for i in 0..4 {
            let expected: f64 = (0..3).map(|k| a[[batch, i, k]] * b[[k]]).sum();
            assert!((result[[batch, i]] - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn scalars_and_invalid_shapes_are_rejected() {
    for (a, b) in [
        (vec![], vec![]),
        (vec![], vec![3]),
        (vec![3], vec![]),
        (vec![2], vec![3]),
        (vec![2, 2, 3], vec![4, 3, 2]),
    ] {
        assert!(std::panic::catch_unwind(|| infer_matmul_shape(&a, &b)).is_err());
        assert!(std::panic::catch_unwind(|| matmul(&values(&a, 1.0), &values(&b, 1.0))).is_err());
    }
}

#[test]
fn empty_dimensions_preserve_shapes_and_gradients() {
    #[trace]
    fn product(a: Tensor, b: Tensor) -> Tensor {
        a.matmul(b)
    }
    let grad = trace_fn::<f64>(product).grad();
    for (sa, sb) in [
        (vec![0], vec![0]),
        (vec![0], vec![2, 0, 3]),
        (vec![0, 2, 3], vec![3]),
        (vec![1, 2, 3], vec![0, 3, 4]),
    ] {
        let a = values(&sa, 1.0);
        let b = values(&sb, 1.0);
        assert_eq!(matmul(&a, &b).shape(), infer_matmul_shape(&sa, &sb));
        let (ga, gb): (ArrayD<f64>, ArrayD<f64>) = grad.eval()((&a, &b));
        assert_eq!(ga.shape(), sa);
        assert_eq!(gb.shape(), sb);
        assert!(ga.iter().chain(gb.iter()).all(|&v| v == 0.0));
    }
}
