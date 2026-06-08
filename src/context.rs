use crate::{Floating, identity::Id, tracing::TensorData};

#[derive(Debug, Clone)]
pub struct Context<D = f32> {
    pub tensors: Vec<Option<TensorData<D>>>,
}

impl<D: Floating> Context<D> {
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    fn ensure_len(&mut self, idx: usize) {
        if self.tensors.len() <= idx {
            self.tensors.resize_with(idx + 1, || None);
        }
    }

    pub fn checked_get(&self, id: &Id) -> &TensorData<D> {
        let idx = id.as_usize();
        self.tensors
            .get(idx)
            .and_then(|t| t.as_ref())
            .unwrap_or_else(|| panic!("tensor({id:?}) was not found in context."))
    }

    pub fn insert(&mut self, id: Id, tensor: TensorData<D>) {
        let idx = id.as_usize();
        self.ensure_len(idx);
        self.tensors[idx] = Some(tensor);
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
