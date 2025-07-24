// Since cudarc 0.11 doesn't support raw kernel launches the same way,
// we'll use a different approach - PTX modules

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use std::sync::Arc;
use crate::{Result, FlameError};

// Helper to allocate from pool and copy data
fn alloc_from_pool_and_copy(device: &Arc<CudaDevice>, data: &[i32]) -> Result<CudaSlice<f32>> {
    let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    let mut cuda_data = crate::tensor::alloc_from_pool(device, f32_data.len())?;
    device.htod_copy_into(&f32_data, &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}


// Helper function for allocating and copying to GPU via memory pool
fn alloc_and_copy_to_pool<T: AsRef<[f32]>>(device: &Arc<CudaDevice>, data: T) -> Result<CudaSlice<f32>> {
    let slice = data.as_ref();
    let mut cuda_data = crate::tensor::alloc_from_pool(device, slice.len())?;
    device.htod_copy_into(slice, &mut cuda_data).map_err(|_| FlameError::CudaDriver)?;
    Ok(cuda_data)
}


// PTX code for our kernels (compiled from CUDA)
const UPDATE_WEIGHTS_PTX: &str = r#"
.version 7.0
.target sm_70
.address_size 64

.visible .entry update_weights_f32(
    .param .u64 weights,
    .param .u64 gradients,
    .param .f32 learning_rate,
    .param .u32 num_elements
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<4>;
    .reg .pred %p<2>;

    ld.param.u64 %rd1, [weights];
    ld.param.u64 %rd2, [gradients];
    ld.param.f32 %f1, [learning_rate];
    ld.param.u32 %r1, [num_elements];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mul.lo.s32 %r2, %r2, %r3;
    mov.u32 %r3, %tid.x;
    add.s32 %r2, %r2, %r3;

    setp.ge.u32 %p1, %r2, %r1;
    @%p1 bra LBB0_2;

    cvt.u64.u32 %rd3, %r2;
    shl.b64 %rd4, %rd3, 2;
    add.s64 %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];
    mul.f32 %f3, %f2, %f1;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];
    sub.f32 %f2, %f2, %f3;
    st.global.f32 [%rd5], %f2;

LBB0_2:
    ret;
}
"#;

pub struct CudaKernels {
    device: Arc<CudaDevice>,
}

impl CudaKernels {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self { device }
    }

    /// Update weights in-place on GPU - CPU fallback for now
    pub fn update_weights(
        &self,
        weights: &mut CudaSlice<f32>,
        gradients: &CudaSlice<f32>,
        learning_rate: f32,
    ) -> Result<()> {
        let num_elements = weights.len();
        if num_elements != gradients.len() {
            return Err(FlameError::InvalidOperation(
                "Weight and gradient tensors must have same size".into()
            ));
        }

        // For now, we'll use a simple CPU implementation
        // In production, we'd compile and load PTX modules
        let mut weights_cpu = self.device.dtoh_sync_copy(weights)
            .map_err(|_| FlameError::CudaDriver)?;
        let gradients_cpu = self.device.dtoh_sync_copy(gradients)
            .map_err(|_| FlameError::CudaDriver)?;

        for i in 0..num_elements {
            weights_cpu[i] -= learning_rate * gradients_cpu[i];
        }

        // Copy back
        *weights = alloc_from_pool_and_copy(&self.device, &weights_cpu)
            .map_err(|_| FlameError::CudaDriver)?;

        Ok(())
    }

    /// Element-wise addition
    pub fn add(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        let num_elements = a.len();
        if num_elements != b.len() {
            return Err(FlameError::InvalidOperation(
                "Tensors must have same size for addition".into()
            ));
        }

        // CPU implementation for now
        let a_cpu = self.device.dtoh_sync_copy(a)
            .map_err(|_| FlameError::CudaDriver)?;
        let b_cpu = self.device.dtoh_sync_copy(b)
            .map_err(|_| FlameError::CudaDriver)?;

        let mut result = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            result[i] = a_cpu[i] + b_cpu[i];
        }

        Ok(alloc_from_pool_and_copy(&self.device, &result)
            .map_err(|_| FlameError::CudaDriver)?)
    }

    /// Element-wise multiplication
    pub fn mul(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        let num_elements = a.len();
        if num_elements != b.len() {
            return Err(FlameError::InvalidOperation(
                "Tensors must have same size for multiplication".into()
            ));
        }

        // CPU implementation for now
        let a_cpu = self.device.dtoh_sync_copy(a)
            .map_err(|_| FlameError::CudaDriver)?;
        let b_cpu = self.device.dtoh_sync_copy(b)
            .map_err(|_| FlameError::CudaDriver)?;

        let mut result = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            result[i] = a_cpu[i] * b_cpu[i];
        }

        Ok(alloc_from_pool_and_copy(&self.device, &result)
            .map_err(|_| FlameError::CudaDriver)?)
    }

    /// Scalar multiplication
    pub fn mul_scalar(
        &self,
        input: &CudaSlice<f32>,
        scalar: f32,
    ) -> Result<CudaSlice<f32>> {
        let num_elements = input.len();

        // CPU implementation for now
        let input_cpu = self.device.dtoh_sync_copy(input)
            .map_err(|_| FlameError::CudaDriver)?;

        let mut result = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            result[i] = input_cpu[i] * scalar;
        }

        Ok(alloc_from_pool_and_copy(&self.device, &result)
            .map_err(|_| FlameError::CudaDriver)?)
    }

    /// ReLU activation
    pub fn relu(
        &self,
        input: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>> {
        let num_elements = input.len();

        // CPU implementation for now
        let input_cpu = self.device.dtoh_sync_copy(input)
            .map_err(|_| FlameError::CudaDriver)?;

        let mut result = vec![0.0f32; num_elements];
        for i in 0..num_elements {
            result[i] = input_cpu[i].max(0.0);
        }

        Ok(alloc_from_pool_and_copy(&self.device, &result)
            .map_err(|_| FlameError::CudaDriver)?)
    }

    /// Fill tensor with value
    pub fn fill(
        &self,
        value: f32,
        num_elements: usize,
    ) -> Result<CudaSlice<f32>> {
        let data = vec![value; num_elements];
        Ok(alloc_from_pool_and_copy(&self.device, &data)
            .map_err(|_| FlameError::CudaDriver)?)
    }
}