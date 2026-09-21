use chainrule::prelude::*;
use ndarray::{Array, ArrayD, arr2};

fn assert_all_close(actual: &ArrayD<f32>, expected: &ArrayD<f32>) {
    assert_eq!(actual.shape(), expected.shape());
    assert!(
        actual
            .iter()
            .zip(expected.iter())
            .all(|(a, e)| (a - e).abs() < 1e-6),
        "actual: {actual:?}\nexpected: {expected:?}"
    );
}

#[test]
fn second_derivative_through_sum() {
    #[trace]
    fn f(x: Tensor) -> Tensor {
        let reduced = x.sum(vec![1], false);
        (reduced * reduced).sum(vec![], false)
    }

    let f = trace_fn::<f32>(f);
    let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let (second,) = f.grad().grad().eval()(&x);

    assert_all_close(&second, &Array::from_elem((2, 3), 6.0).into_dyn());
}

#[test]
fn second_derivative_through_mean() {
    #[trace]
    fn f(x: Tensor) -> Tensor {
        let reduced = x.mean(vec![1], false);
        (reduced * reduced).sum(vec![], false)
    }

    let f = trace_fn::<f32>(f);
    let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let (second,) = f.grad().grad().eval()(&x);

    assert_all_close(&second, &Array::from_elem((2, 3), 2.0 / 3.0).into_dyn());
}

#[test]
fn second_derivative_through_reshape() {
    #[trace]
    fn f(x: Tensor) -> Tensor {
        let reshaped = x.reshape(vec![6]);
        (reshaped * reshaped).sum(vec![], false)
    }

    let f = trace_fn::<f32>(f);
    let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
    let (second,) = f.grad().grad().eval()(&x);

    assert_all_close(&second, &Array::from_elem((2, 3), 2.0).into_dyn());
}

#[test]
fn max_has_zero_second_derivative_away_from_ties() {
    #[trace]
    fn f(x: Tensor) -> Tensor {
        x.max(vec![1], false).sum(vec![], false)
    }

    let f = trace_fn::<f32>(f);
    let x = arr2(&[[1.0, 3.0, 2.0], [6.0, 5.0, 4.0]]).into_dyn();
    let (second,) = f.grad().grad().eval()(&x);

    assert_all_close(&second, &Array::zeros((2, 3)).into_dyn());
}
