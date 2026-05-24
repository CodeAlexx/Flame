#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace at {

struct PhiloxCudaState {
    struct Payload {
        uint64_t val;
        const uint64_t* ptr;

        __host__ __device__ Payload() : val(0), ptr(nullptr) {}
    };

    bool captured_;
    Payload seed_;
    Payload offset_;
    uint64_t offset_intragraph_;

    __host__ __device__ PhiloxCudaState()
        : captured_(false), seed_(), offset_(), offset_intragraph_(0) {}
};

} // namespace at
