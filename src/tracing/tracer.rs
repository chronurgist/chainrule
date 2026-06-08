use crate::{Floating, identity::Id};

use core::ops::{Add, Mul, Neg, Sub};
use std::ops::Div;

pub type TensorData<T = f32> = ndarray::ArrayD<T>;
pub type Tensor = Tracer;

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub struct Tracer {
    id: Id,
}

impl Tracer {
    pub fn new(id: Id) -> Self {
        Self { id }
    }
    pub fn id(&self) -> Id {
        self.id
    }

    pub fn shape(&self) -> Vec<usize> {
        panic!("dummy shape function - only allowed inside #[trace] functions")
    }
}

impl Add for Tracer {
    type Output = Self;
    fn add(self, _rhs: Self) -> Self {
        panic!("dummy Add operator – only allowed inside #[trace] functions")
    }
}

impl Sub for Tracer {
    type Output = Self;
    fn sub(self, _rhs: Self) -> Self {
        panic!("dummy Sub operator – only allowed inside #[trace] functions")
    }
}

impl Mul for Tracer {
    type Output = Self;
    fn mul(self, _rhs: Self) -> Self {
        panic!("dummy Mul operator – only allowed inside #[trace] functions")
    }
}

impl Neg for Tracer {
    type Output = Self;
    fn neg(self) -> Self {
        panic!("dummy Neg operator – only allowed inside #[trace] functions")
    }
}

impl Div for Tracer {
    type Output = Self;

    fn div(self, _: Self) -> Self::Output {
        panic!("dummy Div operator - only allowed inside #[trace] functions")
    }
}

pub trait Item<D: Floating> {
    fn item(&self) -> D;
}

impl<D: Floating> Item<D> for TensorData<D> {
    fn item(&self) -> D {
        if !self.shape().is_empty() {
            panic!(
                "item only works on tensors of 0 dimensions. shape: {:?}",
                self.shape()
            );
        }
        self.first().copied().unwrap()
    }
}
