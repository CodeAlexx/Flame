#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FLAME_CUDA_OK = 0,
  FLAME_CUDA_ERR_INVALID = 1,
  FLAME_CUDA_ERR_UNSUPPORTED = 2,
  FLAME_CUDA_ERR_CUDA = 3
} FlameCudaStatus;

#ifdef __cplusplus
}
#endif
