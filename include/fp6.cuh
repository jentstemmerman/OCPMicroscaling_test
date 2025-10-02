#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Experimental 6-bit float e3m2 (1 sign, 3 exponent, 2 mantissa) dynamic range similar to custom.
// Layout (from MSB to LSB): [s][e2 e1 e0][m1 m0]
// Exponent bias = 3 (so exponent field 0..7 represents -3 .. +4). 0 exponent with non-zero mantissa => subnormal.
// This header provides a software implementation along with wrappers that could be replaced if native hardware intrinsics become available.

struct nv_fp6_e3m2 {
    uint8_t data; // lower 6 bits used
    __host__ __device__ nv_fp6_e3m2() : data(0) {}
    __host__ __device__ explicit nv_fp6_e3m2(uint8_t v) : data(v & 0x3Fu) {}
};

__host__ __device__ inline nv_fp6_e3m2 float_to_fp6(float x);
__host__ __device__ inline float fp6_to_float(nv_fp6_e3m2 v);

// Vectorized helpers for arrays
void convert_fp32_to_fp6(const float* d_in, nv_fp6_e3m2* d_out, size_t n, cudaStream_t stream = 0);
void convert_fp6_to_fp32(const nv_fp6_e3m2* d_in, float* d_out, size_t n, cudaStream_t stream = 0);

// Kernel launch grid helper
inline dim3 makeGrid(size_t n, int block=256) { return dim3( (unsigned)((n + block - 1)/block) ); }
