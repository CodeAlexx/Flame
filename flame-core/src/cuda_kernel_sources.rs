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

pub const DIV_KERNEL: &str = r#"
extern "C" __global__ void div_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
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

// Permutation kernel: NHWC to NCHW format conversion
// [batch, height, width, channels] -> [batch, channels, height, width]
pub const PERMUTE_NHWC_TO_NCHW_KERNEL: &str = r#"
extern "C" __global__ void permute_nhwc_to_nchw_kernel(
    const float* input,
    float* output,
    int batch,
    int height,
    int width,
    int channels
) {
    int total_elements = batch * height * width * channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Decompose linear index into NHWC coordinates
        int b = idx / (height * width * channels);
        int remainder = idx % (height * width * channels);
        int h = remainder / (width * channels);
        remainder = remainder % (width * channels);
        int w = remainder / channels;
        int c = remainder % channels;
        
        // Calculate target index in NCHW format
        int target_idx = b * (channels * height * width) +
                        c * (height * width) +
                        h * width +
                        w;
        
        // Copy the element
        output[target_idx] = input[idx];
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

// Elementwise exponential
pub const EXP_KERNEL: &str = r#"
extern "C" __global__ void exp_kernel(
    const float* input,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(input[idx]);
    }
}
"#;

// Elementwise natural logarithm (numerically safe)
pub const LOG_KERNEL: &str = r#"
extern "C" __global__ void log_kernel(
    const float* input,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = input[idx];
        if (v < 1e-20f) v = 1e-20f;
        out[idx] = logf(v);
    }
}
"#;

// Elementwise max
pub const MAX_ELEMWISE_KERNEL: &str = r#"
extern "C" __global__ void max_elemwise_kernel(
    const float* a,
    const float* b,
    float* out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float av = a[idx];
        float bv = b[idx];
        out[idx] = av > bv ? av : bv;
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

pub const POW_KERNEL: &str = r#"
extern "C" __global__ void pow_kernel(
    const float* input,
    float* output,
    float exponent,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = powf(input[idx], exponent);
    }
}
"#;

pub const SIN_KERNEL: &str = r#"
extern "C" __global__ void sin_kernel(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sinf(input[idx]);
    }
}
"#;

pub const COS_KERNEL: &str = r#"
extern "C" __global__ void cos_kernel(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = cosf(input[idx]);
    }
}
"#;

pub const SQRT_KERNEL: &str = r#"
extern "C" __global__ void sqrt_kernel(
    const float* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = sqrtf(input[idx]);
    }
}
"#;

pub const NARROW_KERNEL: &str = r#"
extern "C" __global__ void narrow_kernel(
    const float* input,
    float* output,
    int input_size0, int input_size1, int input_size2, int input_size3,
    int output_size0, int output_size1, int output_size2, int output_size3,
    int dim, int start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_size = output_size0 * output_size1 * output_size2 * output_size3;
    
    if (idx >= total_output_size) return;
    
    // Calculate output indices
    int out_idx3 = idx % output_size3;
    int out_idx2 = (idx / output_size3) % output_size2;
    int out_idx1 = (idx / (output_size3 * output_size2)) % output_size1;
    int out_idx0 = idx / (output_size3 * output_size2 * output_size1);
    
    // Calculate input indices
    int in_idx0 = out_idx0;
    int in_idx1 = out_idx1;
    int in_idx2 = out_idx2;
    int in_idx3 = out_idx3;
    
    // Adjust the index for the narrowed dimension
    if (dim == 0) {
        in_idx0 += start;
    } else if (dim == 1) {
        in_idx1 += start;
    } else if (dim == 2) {
        in_idx2 += start;
    } else if (dim == 3) {
        in_idx3 += start;
    }
    
    // Calculate flat index for input
    int input_idx = in_idx0 * (input_size1 * input_size2 * input_size3) +
                    in_idx1 * (input_size2 * input_size3) +
                    in_idx2 * input_size3 +
                    in_idx3;
    
    output[idx] = input[input_idx];
}
"#;

// Generic N-D index_select along a single dimension
pub const INDEX_SELECT_KERNEL: &str = r#"
extern "C" __global__ void index_select_kernel(
    const float* input,
    const float* indices, // indices as float, cast to int
    float* output,
    int ndim,
    const int* in_dims,
    const int* out_dims,
    const int* in_strides,
    const int* out_strides,
    int select_dim,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int rem = idx;
    int coords[8];
    for (int d = 0; d < ndim; ++d) {
        int stride = out_strides[d];
        int c = rem / stride;
        rem = rem % stride;
        coords[d] = c;
    }

    int sel_idx = (int)indices[coords[select_dim]];
    if (sel_idx < 0) sel_idx = 0;
    if (sel_idx >= in_dims[select_dim]) sel_idx = in_dims[select_dim] - 1;

    int in_index = 0;
    for (int d = 0; d < ndim; ++d) {
        int c = (d == select_dim) ? sel_idx : coords[d];
        in_index += c * in_strides[d];
    }
    output[idx] = input[in_index];
}
"#;

// Generic N-D slice with per-dimension start offsets
pub const SLICE_KERNEL: &str = r#"
extern "C" __global__ void slice_kernel(
    const float* input,
    float* output,
    int ndim,
    const int* in_strides,
    const int* out_strides,
    const int* starts,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    int rem = idx;
    int coords[8];
    for (int d = 0; d < ndim; ++d) {
        int stride = out_strides[d];
        int c = rem / stride;
        rem = rem % stride;
        coords[d] = c;
    }

    int in_index = 0;
    for (int d = 0; d < ndim; ++d) {
        int c = starts[d] + coords[d];
        in_index += c * in_strides[d];
    }
    output[idx] = input[in_index];
}
"#;

// Sum along a dimension, keeping that dimension (size 1)
pub const SUM_DIM_KEEPDIM_KERNEL: &str = r#"
extern "C" __global__ void sum_dim_keepdim_kernel(
    const float* input,
    float* output,
    const float* dims_f32,
    int ndim,
    int reduce_dim,
    int out_elems
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_elems) return;

    // Load dims from f32
    int dims[8];
    for (int i = 0; i < ndim && i < 8; ++i) dims[i] = (int)dims_f32[i];

    // Compute strides
    int strides[8];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Reconstruct coordinates for output (same rank, reduce_dim = 1-sized)
    int rem = tid;
    int out_coords[8];
    for (int i = 0; i < ndim; ++i) {
        int size = (i == reduce_dim) ? 1 : dims[i];
        int stride = 1;
        for (int j = i + 1; j < ndim; ++j) {
            stride *= (j == reduce_dim) ? 1 : dims[j];
        }
        out_coords[i] = (size == 0) ? 0 : (rem / stride) % size;
    }

    // Base index into input
    int base_idx = 0;
    for (int i = 0; i < ndim; ++i) {
        int coord = (i == reduce_dim) ? 0 : out_coords[i];
        base_idx += coord * strides[i];
    }

    float sum = 0.0f;
    for (int d = 0; d < dims[reduce_dim]; ++d) {
        int idx = base_idx + d * strides[reduce_dim];
        sum += input[idx];
    }
    output[tid] = sum;
}
"#;

// NCHW -> NHWC permutation kernel
pub const PERMUTE_NCHW_TO_NHWC_KERNEL: &str = r#"
extern "C" __global__ void permute_nchw_to_nhwc_kernel(
    const float* input,
    float* output,
    int batch,
    int channels,
    int height,
    int width
) {
    int total = batch * height * width * channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int b = idx / (height * width * channels);
        int rem = idx % (height * width * channels);
        int h = rem / (width * channels);
        rem = rem % (width * channels);
        int w = rem / channels;
        int c = rem % channels;
        int in_idx = b * (channels * height * width) + c * (height * width) + h * width + w;
        output[idx] = input[in_idx];
    }
}
"#;

// Weight permutation kernels
// [KH,KW,IC,OC] -> [OC,IC,KH,KW]
pub const PERMUTE_W_KH_KW_IC_OC_TO_OC_IC_KH_KW: &str = r#"
extern "C" __global__ void permute_w_khwkicoc_to_ocickhkw(
    const float* input,
    float* output,
    int kh, int kw, int ic, int oc
) {
    int total = kh * kw * ic * oc;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int tmp = idx;
    int oc_i = tmp % oc; tmp /= oc;
    int ic_i = tmp % ic; tmp /= ic;
    int kw_i = tmp % kw; tmp /= kw;
    int kh_i = tmp;
    int in_idx = ((kh_i * kw + kw_i) * ic + ic_i) * oc + oc_i;
    int out_idx = ((oc_i * ic + ic_i) * kh + kh_i) * kw + kw_i;
    output[out_idx] = input[in_idx];
}
"#;

// [OC,IC,KH,KW] -> [KH,KW,IC,OC]
pub const PERMUTE_W_OC_IC_KH_KW_TO_KH_KW_IC_OC: &str = r#"
extern "C" __global__ void permute_w_ocickhkw_to_khwkicoc(
    const float* input,
    float* output,
    int oc, int ic, int kh, int kw
) {
    int total = kh * kw * ic * oc;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int tmp = idx;
    int oc_i = tmp % oc; tmp /= oc;
    int ic_i = tmp % ic; tmp /= ic;
    int kw_i = tmp % kw; tmp /= kw;
    int kh_i = tmp;
    int in_idx = ((oc_i * ic + ic_i) * kh + kh_i) * kw + kw_i;
    int out_idx = ((kh_i * kw + kw_i) * ic + ic_i) * oc + oc_i;
    output[out_idx] = input[in_idx];
}
"#;

/// Helper to compile CUDA C to PTX
/// DEPRECATED: Use cuda_kernel_compiler::compile_cuda_kernel instead
pub fn compile_cuda_kernel(source: &str) -> Result<cudarc::nvrtc::Ptx, Box<dyn std::error::Error>> {
    // Delegate to real implementation
    use crate::cuda_kernel_compiler;
    cuda_kernel_compiler::compile_cuda_kernel(source, "kernel")
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

// NHWC image ops: resize bilinear
pub const RESIZE_BILINEAR_NHWC_KERNEL: &str = r#"
extern "C" __global__ void resize_bilinear_nhwc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int H_in, int W_in, int C,
    int H_out, int W_out,
    int align_corners
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out * C;
    if (idx >= total) return;
    int c = idx % C;
    int w = (idx / C) % W_out;
    int h = (idx / (C * W_out)) % H_out;
    int n = idx / (C * W_out * H_out);
    float h_scale = align_corners ? (float)(H_in - 1) / (H_out - 1) : (float)H_in / H_out;
    float w_scale = align_corners ? (float)(W_in - 1) / (W_out - 1) : (float)W_in / W_out;
    float h_idx = align_corners ? h * h_scale : (h + 0.5f) * h_scale - 0.5f;
    float w_idx = align_corners ? w * w_scale : (w + 0.5f) * w_scale - 0.5f;
    int h0 = (int)floorf(h_idx);
    int w0 = (int)floorf(w_idx);
    int h1 = h0 + 1;
    int w1 = w0 + 1;
    h0 = max(0, min(h0, H_in - 1));
    w0 = max(0, min(w0, W_in - 1));
    h1 = max(0, min(h1, H_in - 1));
    w1 = max(0, min(w1, W_in - 1));
    float h_frac = h_idx - floorf(h_idx);
    float w_frac = w_idx - floorf(w_idx);
    int base_in = n * (H_in * W_in * C);
    int idx00 = base_in + (h0 * W_in + w0) * C + c;
    int idx01 = base_in + (h0 * W_in + w1) * C + c;
    int idx10 = base_in + (h1 * W_in + w0) * C + c;
    int idx11 = base_in + (h1 * W_in + w1) * C + c;
    float v00 = input[idx00];
    float v01 = input[idx01];
    float v10 = input[idx10];
    float v11 = input[idx11];
    float v0 = v00 * (1.0f - w_frac) + v01 * w_frac;
    float v1 = v10 * (1.0f - w_frac) + v11 * w_frac;
    output[idx] = v0 * (1.0f - h_frac) + v1 * h_frac;
}
"#;

// NHWC image ops: center crop
pub const CENTER_CROP_NHWC_KERNEL: &str = r#"
extern "C" __global__ void center_crop_nhwc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int H, int W, int C,
    int y0, int x0, int tgt_h, int tgt_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * tgt_h * tgt_w * C;
    if (idx >= total) return;
    int c = idx % C;
    int x = (idx / C) % tgt_w;
    int y = (idx / (C * tgt_w)) % tgt_h;
    int n = idx / (C * tgt_w * tgt_h);
    int src_y = min(max(y0 + y, 0), H - 1);
    int src_x = min(max(x0 + x, 0), W - 1);
    int in_idx = n * (H * W * C) + (src_y * W + src_x) * C + c;
    output[idx] = input[in_idx];
}
"#;

// NHWC image ops: per-channel normalize
pub const NORMALIZE_NHWC_KERNEL: &str = r#"
extern "C" __global__ void normalize_nhwc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    int N, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;
    int c = idx % C;
    output[idx] = (input[idx] - mean[c]) * inv_std[c];
}
"#;
