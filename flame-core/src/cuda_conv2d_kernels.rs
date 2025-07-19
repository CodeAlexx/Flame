//! CUDA kernels for Conv2D forward and backward operations
//! Implements im2col/col2im for efficient convolution

pub const CONV2D_KERNELS: &str = r#"
// Simplified im2col for common case (stride=1, padding=1)
extern "C" __global__ void im2col_kernel_simple(
    const float* data_im,
    float* data_col,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int out_h,
    const int out_w
) {
    const int pad_h = 1;
    const int pad_w = 1;
    const int stride_h = 1;
    const int stride_w = 1;
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * out_h * out_w * kernel_h * kernel_w;
    
    if (index >= total_size) return;
    
    // Decompose index
    int w_col = index % out_w;
    int h_col = (index / out_w) % out_h;
    int c_im = (index / (out_w * out_h)) % channels;
    int kw = (index / (out_w * out_h * channels)) % kernel_w;
    int kh = (index / (out_w * out_h * channels * kernel_w)) % kernel_h;
    int batch = index / (out_w * out_h * channels * kernel_w * kernel_h);
    
    // Calculate position in input image
    int h_im = h_col * stride_h - pad_h + kh;
    int w_im = w_col * stride_w - pad_w + kw;
    
    // Calculate position in col buffer
    int col_index = batch * (channels * kernel_h * kernel_w * out_h * out_w) +
                    (c_im * kernel_h * kernel_w + kh * kernel_w + kw) * (out_h * out_w) +
                    h_col * out_w + w_col;
    
    // Boundary check and copy
    if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
        int im_index = batch * (channels * height * width) +
                       c_im * (height * width) +
                       h_im * width + w_im;
        data_col[col_index] = data_im[im_index];
    } else {
        data_col[col_index] = 0.0f;
    }
}

// Full im2col with all parameters (split into two kernels to reduce param count)
extern "C" __global__ void im2col_kernel(
    const float* data_im,
    float* data_col,
    const int* dims,  // Array containing: batch_size, channels, height, width, kernel_h, kernel_w
    const int* conv_params  // Array containing: pad_h, pad_w, stride_h, stride_w, out_h, out_w
) {
    const int batch_size = dims[0];
    const int channels = dims[1];
    const int height = dims[2];
    const int width = dims[3];
    const int kernel_h = dims[4];
    const int kernel_w = dims[5];
    
    const int pad_h = conv_params[0];
    const int pad_w = conv_params[1];
    const int stride_h = conv_params[2];
    const int stride_w = conv_params[3];
    const int out_h = conv_params[4];
    const int out_w = conv_params[5];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * out_h * out_w * kernel_h * kernel_w;
    
    if (index >= total_size) return;
    
    // Decompose index
    int w_col = index % out_w;
    int h_col = (index / out_w) % out_h;
    int c_im = (index / (out_w * out_h)) % channels;
    int kw = (index / (out_w * out_h * channels)) % kernel_w;
    int kh = (index / (out_w * out_h * channels * kernel_w)) % kernel_h;
    int batch = index / (out_w * out_h * channels * kernel_w * kernel_h);
    
    // Calculate position in input image
    int h_im = h_col * stride_h - pad_h + kh;
    int w_im = w_col * stride_w - pad_w + kw;
    
    // Calculate position in col buffer
    int col_index = batch * (channels * kernel_h * kernel_w * out_h * out_w) +
                    (c_im * kernel_h * kernel_w + kh * kernel_w + kw) * (out_h * out_w) +
                    h_col * out_w + w_col;
    
    // Boundary check and copy
    if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
        int im_index = batch * (channels * height * width) +
                       c_im * (height * width) +
                       h_im * width + w_im;
        data_col[col_index] = data_im[im_index];
    } else {
        data_col[col_index] = 0.0f;
    }
}

// Simplified col2im for common case (stride=1, padding=1)
extern "C" __global__ void col2im_kernel_simple(
    const float* data_col,
    float* data_im,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int out_h,
    const int out_w
) {
    const int pad_h = 1;
    const int pad_w = 1;
    const int stride_h = 1;
    const int stride_w = 1;
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * height * width;
    
    if (index >= total_size) return;
    
    // Decompose index for output image
    int w_im = index % width;
    int h_im = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int batch = index / (width * height * channels);
    
    float val = 0.0f;
    
    // Iterate over all windows that contain this pixel
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            // Calculate which output positions could have used this input pixel
            int h_col_start = (h_im + pad_h - kh) / stride_h;
            int w_col_start = (w_im + pad_w - kw) / stride_w;
            
            // Check if the position is valid
            if (h_col_start * stride_h == h_im + pad_h - kh &&
                w_col_start * stride_w == w_im + pad_w - kw &&
                h_col_start >= 0 && h_col_start < out_h &&
                w_col_start >= 0 && w_col_start < out_w) {
                
                // Calculate position in col buffer
                int col_index = batch * (channels * kernel_h * kernel_w * out_h * out_w) +
                               (c * kernel_h * kernel_w + kh * kernel_w + kw) * (out_h * out_w) +
                               h_col_start * out_w + w_col_start;
                
                val += data_col[col_index];
            }
        }
    }
    
    data_im[index] = val;
}

extern "C" __global__ void col2im_kernel(
    const float* data_col,
    float* data_im,
    const int* dims,  // Array containing: batch_size, channels, height, width, kernel_h, kernel_w
    const int* conv_params  // Array containing: pad_h, pad_w, stride_h, stride_w, out_h, out_w
) {
    const int batch_size = dims[0];
    const int channels = dims[1];
    const int height = dims[2];
    const int width = dims[3];
    const int kernel_h = dims[4];
    const int kernel_w = dims[5];
    
    const int pad_h = conv_params[0];
    const int pad_w = conv_params[1];
    const int stride_h = conv_params[2];
    const int stride_w = conv_params[3];
    const int out_h = conv_params[4];
    const int out_w = conv_params[5];
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * height * width;
    
    if (index >= total_size) return;
    
    // Decompose index for output image
    int w_im = index % width;
    int h_im = (index / width) % height;
    int c = (index / (width * height)) % channels;
    int batch = index / (width * height * channels);
    
    float val = 0.0f;
    
    // Iterate over all windows that contain this pixel
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            // Calculate which output positions could have used this input pixel
            int h_col_start = (h_im + pad_h - kh) / stride_h;
            int w_col_start = (w_im + pad_w - kw) / stride_w;
            
            // Check if the position is valid
            if (h_col_start * stride_h == h_im + pad_h - kh &&
                w_col_start * stride_w == w_im + pad_w - kw &&
                h_col_start >= 0 && h_col_start < out_h &&
                w_col_start >= 0 && w_col_start < out_w) {
                
                // Calculate position in col buffer
                int col_index = batch * (channels * kernel_h * kernel_w * out_h * out_w) +
                               (c * kernel_h * kernel_w + kh * kernel_w + kw) * (out_h * out_w) +
                               h_col_start * out_w + w_col_start;
                
                val += data_col[col_index];
            }
        }
    }
    
    data_im[index] = val;
}

extern "C" __global__ void add_bias_nhwc_kernel(
    float* output,
    const float* bias,
    const int batch_size,
    const int height,
    const int width,
    const int channels
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * height * width * channels;
    
    if (index >= total_size) return;
    
    int c = index % channels;
    output[index] += bias[c];
}

extern "C" __global__ void add_bias_nchw_kernel(
    float* output,
    const float* bias,
    const int batch_size,
    const int channels,
    const int spatial_size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * spatial_size;
    
    if (index >= total_size) return;
    
    int c = (index / spatial_size) % channels;
    output[index] += bias[c];
}

extern "C" __global__ void bias_grad_kernel(
    const float* grad_output,
    float* grad_bias,
    const int batch_size,
    const int channels,
    const int spatial_size
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (c >= channels) return;
    
    float sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < spatial_size; ++s) {
            int idx = b * channels * spatial_size + c * spatial_size + s;
            sum += grad_output[idx];
        }
    }
    
    grad_bias[c] = sum;
}

// Optimized im2col using shared memory
extern "C" __global__ void im2col_optimized_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int out_h,
    const int out_w
) {
    extern __shared__ float smem[];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int blocks_per_row = (out_w + block_size - 1) / block_size;
    
    const int block_row = blockIdx.x / blocks_per_row;
    const int block_col = blockIdx.x % blocks_per_row;
    const int batch = blockIdx.y;
    const int channel = blockIdx.z;
    
    const int h_col = block_row;
    const int w_col_start = block_col * block_size;
    const int w_col = w_col_start + tid;
    
    if (h_col >= out_h || w_col >= out_w) return;
    
    // Load a tile of input data into shared memory
    const int tile_h = kernel_h;
    const int tile_w = kernel_w + (block_size - 1) * stride_w;
    const int h_im_start = h_col * stride_h - pad_h;
    const int w_im_start = w_col_start * stride_w - pad_w;
    
    // Cooperative loading into shared memory
    for (int i = tid; i < tile_h * tile_w; i += block_size) {
        int local_h = i / tile_w;
        int local_w = i % tile_w;
        int h_im = h_im_start + local_h;
        int w_im = w_im_start + local_w;
        
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) {
            int im_idx = batch * channels * height * width +
                        channel * height * width +
                        h_im * width + w_im;
            smem[i] = data_im[im_idx];
        } else {
            smem[i] = 0.0f;
        }
    }
    
    __syncthreads();
    
    // Each thread processes one output column
    if (w_col < out_w) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int smem_h = kh;
                int smem_w = tid * stride_w + kw;
                float val = smem[smem_h * tile_w + smem_w];
                
                int col_idx = batch * channels * kernel_h * kernel_w * out_h * out_w +
                             (channel * kernel_h * kernel_w + kh * kernel_w + kw) * out_h * out_w +
                             h_col * out_w + w_col;
                
                data_col[col_idx] = val;
            }
        }
    }
}

// Error checking helper
extern "C" __global__ void check_conv_dimensions_kernel(
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_h,
    const int in_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    int* error_flag
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Calculate expected output dimensions
        int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
        
        // Check for valid dimensions
        if (out_h <= 0 || out_w <= 0) {
            *error_flag = 1;  // Invalid output dimensions
        } else if (kernel_h > in_h + 2 * pad_h || kernel_w > in_w + 2 * pad_w) {
            *error_flag = 2;  // Kernel larger than padded input
        } else {
            *error_flag = 0;  // All good
        }
    }
}
"#;