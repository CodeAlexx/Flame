//! CUDA C kernel sources that will be compiled to PTX at runtime

pub const ADD_KERNEL: &str = r#"
extern "C" __global__ void add_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}
"#;

pub const MUL_KERNEL: &str = r#"
extern "C" __global__ void mul_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}
"#;

pub const MUL_SCALAR_KERNEL: &str = r#"
extern "C" __global__ void mul_scalar_kernel(
    const float* input,
    float scalar,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = input[idx] * scalar;
    }
}
"#;

pub const ADD_SCALAR_KERNEL: &str = r#"
extern "C" __global__ void add_scalar_kernel(
    const float* input,
    float scalar,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = input[idx] + scalar;
    }
}
"#;

pub const RELU_KERNEL: &str = r#"
extern "C" __global__ void relu_kernel(
    const float* input,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}
"#;

pub const SIGMOID_KERNEL: &str = r#"
extern "C" __global__ void sigmoid_kernel(
    const float* input,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        out[idx] = 1.0f / (1.0f + expf(-val));
    }
}
"#;

pub const GELU_KERNEL: &str = r#"
extern "C" __global__ void gelu_kernel(
    const float* input,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
        out[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}
"#;

pub const SILU_KERNEL: &str = r#"
extern "C" __global__ void silu_kernel(
    const float* input,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        out[idx] = x / (1.0f + expf(-x));
    }
}
"#;

pub const TANH_KERNEL: &str = r#"
extern "C" __global__ void tanh_kernel(
    const float* input,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(input[idx]);
    }
}
"#;

pub const SUM_KERNEL: &str = r#"
extern "C" __global__ void sum_kernel(
    const float* input,
    float* out,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (idx < n) {
        sum = input[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}
"#;

pub const TRANSPOSE_KERNEL: &str = r#"
extern "C" __global__ void transpose_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < cols && y < rows) {
        int in_idx = y * cols + x;
        int out_idx = x * rows + y;
        output[out_idx] = input[in_idx];
    }
}
"#;

pub const UPDATE_WEIGHTS_KERNEL: &str = r#"
extern "C" __global__ void update_weights_kernel(
    float* weights,
    const float* gradients,
    float lr,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= lr * gradients[idx];
    }
}
"#;

pub const LEAKY_RELU_KERNEL: &str = r#"
extern "C" __global__ void leaky_relu_kernel(
    const float* input,
    float* out,
    float negative_slope,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        out[idx] = val >= 0.0f ? val : negative_slope * val;
    }
}
"#;

pub const ELU_KERNEL: &str = r#"
extern "C" __global__ void elu_kernel(
    const float* input,
    float* out,
    float alpha,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        out[idx] = val >= 0.0f ? val : alpha * (expf(val) - 1.0f);
    }
}
"#;

/// Helper to compile CUDA C to PTX
pub fn compile_cuda_kernel(source: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // For now, return the source as bytes - real implementation would use NVRTC
    // This is a placeholder that assumes pre-compiled PTX
    Ok(source.as_bytes().to_vec())
}