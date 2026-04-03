#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
pub enum GradStorePolicy {
    InternalFP32_PublicBF16,
}

impl Default for GradStorePolicy {
    fn default() -> Self {
        Self::InternalFP32_PublicBF16
    }
}
