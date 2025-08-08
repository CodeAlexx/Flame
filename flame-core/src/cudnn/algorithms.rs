// cuDNN Algorithm Selection
// Provides intelligent algorithm selection for optimal convolution performance

use std::os::raw::c_int;

// cuDNN convolution forward algorithms
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: c_int = 0;
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM: c_int = 1;
pub const CUDNN_CONVOLUTION_FWD_ALGO_GEMM: c_int = 2;
pub const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT: c_int = 3;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT: c_int = 4;
pub const CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING: c_int = 5;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD: c_int = 6;
pub const CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED: c_int = 7;

/// Algorithm selector for convolution operations
pub struct AlgorithmSelector;

impl AlgorithmSelector {
    /// Select the best algorithm based on kernel size and input dimensions
    pub fn select_forward_algorithm(
        kernel_h: usize,
        kernel_w: usize,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> c_int {
        // Winograd is fastest for 3x3 kernels with specific conditions
        // But it's very picky about tensor formats and dimensions
        if kernel_h == 3 && kernel_w == 3 {
            // Winograd works best with:
            // - Batch size >= 1
            // - Channels that are multiples of 8 (and preferably >= 64)
            // - Spatial dimensions >= 8 and multiples of 8
            // - No stride or stride == 1
            if channels % 8 == 0 && channels >= 64 && 
               height >= 8 && width >= 8 &&
               height % 8 == 0 && width % 8 == 0 {
                // Still might fail, so we'll handle fallback in conv2d
                return CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
            }
        }
        
        // For 1x1 kernels, IMPLICIT_GEMM is optimal
        if kernel_h == 1 && kernel_w == 1 {
            return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        }
        
        // For 5x5 kernels, consider FFT for large spatial dimensions
        if kernel_h == 5 && kernel_w == 5 && height >= 128 && width >= 128 {
            return CUDNN_CONVOLUTION_FWD_ALGO_FFT;
        }
        
        // For very large kernels, use GEMM
        if kernel_h > 5 || kernel_w > 5 {
            return CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
        }
        
        // Default to IMPLICIT_PRECOMP_GEMM for general cases
        // This is more efficient than IMPLICIT_GEMM for most cases
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    }
    
    /// Get a fallback algorithm if the primary choice fails
    pub fn get_fallback_algorithm(failed_algo: c_int) -> c_int {
        match failed_algo {
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => {
                // If Winograd fails, try IMPLICIT_PRECOMP_GEMM
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            },
            CUDNN_CONVOLUTION_FWD_ALGO_FFT => {
                // If FFT fails, fall back to GEMM
                CUDNN_CONVOLUTION_FWD_ALGO_GEMM
            },
            _ => {
                // For all other failures, use the most compatible algorithm
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            }
        }
    }
    
    /// Get algorithm name for logging
    pub fn algorithm_name(algo: c_int) -> &'static str {
        match algo {
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => "IMPLICIT_GEMM",
            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => "IMPLICIT_PRECOMP_GEMM",
            CUDNN_CONVOLUTION_FWD_ALGO_GEMM => "GEMM",
            CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => "DIRECT",
            CUDNN_CONVOLUTION_FWD_ALGO_FFT => "FFT",
            CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => "FFT_TILING",
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => "WINOGRAD",
            CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => "WINOGRAD_NONFUSED",
            _ => "UNKNOWN",
        }
    }
}