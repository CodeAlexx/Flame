extern "C" {

// Weight update kernel - the core operation for training
__global__ void update_weights_f32(
    float* weights,
    const float* gradients,
    float learning_rate,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// Element-wise addition
__global__ void add_f32(
    const float* a,
    const float* b,
    float* out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = a[idx] + b[idx];
    }
}

// Element-wise multiplication
__global__ void mul_f32(
    const float* a,
    const float* b,
    float* out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = a[idx] * b[idx];
    }
}

// Scalar multiplication
__global__ void mul_scalar_f32(
    const float* input,
    float scalar,
    float* out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = input[idx] * scalar;
    }
}

// ReLU activation
__global__ void relu_f32(
    const float* input,
    float* out,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU backward
__global__ void relu_backward_f32(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
    }
}

// Squared loss (for simple testing)
__global__ void mse_loss_f32(
    const float* predictions,
    const float* targets,
    float* loss,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for reduction
    extern __shared__ float sdata[];
    
    float sum = 0.0f;
    if (idx < num_elements) {
        float diff = predictions[idx] - targets[idx];
        sum = diff * diff;
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (threadIdx.x == 0) {
        atomicAdd(loss, sdata[0] / num_elements);
    }
}

// MSE backward
__global__ void mse_backward_f32(
    const float* predictions,
    const float* targets,
    float* grad_predictions,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        grad_predictions[idx] = 2.0f * (predictions[idx] - targets[idx]) / num_elements;
    }
}

// Fill tensor with value
__global__ void fill_f32(
    float* tensor,
    float value,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor[idx] = value;
    }
}

// Copy kernel (for contiguous copy)
__global__ void copy_f32(
    const float* src,
    float* dst,
    int num_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        dst[idx] = src[idx];
    }
}

} // extern "C"