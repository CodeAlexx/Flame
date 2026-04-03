#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Elementwise binary in-place kernels (dst = Op(dst, src))
template <typename T, typename Op>
__global__ void inplace_binary_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = Op::apply(dst[idx], src[idx]);
    }
}

struct AddOp {
    template <typename T>
    __device__ static inline T apply(T a, T b) {
        return a + b;
    }
};

struct MulOp {
    template <typename T>
    __device__ static inline T apply(T a, T b) {
        return a * b;
    }
};

template <typename T, typename Op>
static inline void launch_inplace_impl(
    T* dst,
    const T* src,
    long long n,
    cudaStream_t stream) {
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    inplace_binary_kernel<T, Op><<<grid, block, 0, stream>>>(dst, src, n);
}

extern "C" void launch_add_inplace_f32(
    float* dst,
    const float* src,
    long long n,
    cudaStream_t stream) {
    launch_inplace_impl<float, AddOp>(dst, src, n, stream);
}

extern "C" void launch_add_inplace_bf16(
    void* dst,
    const void* src,
    long long n,
    cudaStream_t stream) {
    launch_inplace_impl<__nv_bfloat16, AddOp>(
        static_cast<__nv_bfloat16*>(dst),
        static_cast<const __nv_bfloat16*>(src),
        n,
        stream);
}

extern "C" void launch_mul_inplace_f32(
    float* dst,
    const float* src,
    long long n,
    cudaStream_t stream) {
    launch_inplace_impl<float, MulOp>(dst, src, n, stream);
}

extern "C" void launch_mul_inplace_bf16(
    void* dst,
    const void* src,
    long long n,
    cudaStream_t stream) {
    launch_inplace_impl<__nv_bfloat16, MulOp>(
        static_cast<__nv_bfloat16*>(dst),
        static_cast<const __nv_bfloat16*>(src),
        n,
        stream);
}

// Scalar transform helpers (dst = Op(src, scalar))

template <typename T>
__device__ inline T add_scalar_convert(T value, float scalar) {
    return value + static_cast<T>(scalar);
}

template <>
__device__ inline __nv_bfloat16 add_scalar_convert(__nv_bfloat16 value, float scalar) {
    return __float2bfloat16(__bfloat162float(value) + scalar);
}

template <typename T>
__device__ inline T mul_scalar_convert(T value, float scalar) {
    return value * static_cast<T>(scalar);
}

template <>
__device__ inline __nv_bfloat16 mul_scalar_convert(__nv_bfloat16 value, float scalar) {
    return __float2bfloat16(__bfloat162float(value) * scalar);
}

template <typename T, typename Op>
__global__ void scalar_transform_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    float scalar,
    long long n) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = Op::apply(src[idx], scalar);
    }
}

template <typename T>
struct AddScalarOp {
    __device__ static inline T apply(T value, float scalar) {
        return add_scalar_convert(value, scalar);
    }
};

template <typename T>
struct MulScalarOp {
    __device__ static inline T apply(T value, float scalar) {
        return mul_scalar_convert(value, scalar);
    }
};

template <typename T, typename Op>
static inline void launch_scalar_transform_impl(
    T* dst,
    const T* src,
    float scalar,
    long long n,
    cudaStream_t stream) {
    const int block = 256;
    const int grid = static_cast<int>((n + block - 1) / block);
    scalar_transform_kernel<T, Op><<<grid, block, 0, stream>>>(dst, src, scalar, n);
}

extern "C" void launch_mul_scalar_f32(
    float* dst,
    const float* src,
    float scalar,
    long long n,
    cudaStream_t stream) {
    launch_scalar_transform_impl<float, MulScalarOp<float>>(dst, src, scalar, n, stream);
}

extern "C" void launch_mul_scalar_bf16(
    void* dst,
    const void* src,
    float scalar,
    long long n,
    cudaStream_t stream) {
    launch_scalar_transform_impl<__nv_bfloat16, MulScalarOp<__nv_bfloat16>>(
        static_cast<__nv_bfloat16*>(dst),
        static_cast<const __nv_bfloat16*>(src),
        scalar,
        n,
        stream);
}

extern "C" void launch_add_scalar_f32(
    float* dst,
    const float* src,
    float scalar,
    long long n,
    cudaStream_t stream) {
    launch_scalar_transform_impl<float, AddScalarOp<float>>(dst, src, scalar, n, stream);
}

extern "C" void launch_add_scalar_bf16(
    void* dst,
    const void* src,
    float scalar,
    long long n,
    cudaStream_t stream) {
    launch_scalar_transform_impl<__nv_bfloat16, AddScalarOp<__nv_bfloat16>>(
        static_cast<__nv_bfloat16*>(dst),
        static_cast<const __nv_bfloat16*>(src),
        scalar,
        n,
        stream);
}
