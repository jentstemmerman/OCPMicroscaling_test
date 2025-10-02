#include "fp6.cuh"
#include <cmath>
#include <cstdio>

// Conversion constants
static constexpr int FP6_EXP_BITS = 3;
static constexpr int FP6_MAN_BITS = 2;
static constexpr int FP6_EXP_BIAS = 3; // bias = 2^(k-1)-1 = 3 for k=3
static constexpr int FP6_EXP_MAX = (1<<FP6_EXP_BITS)-1; // 7
static constexpr int FP6_MAN_MAX = (1<<FP6_MAN_BITS)-1; // 3

__host__ __device__ inline nv_fp6_e3m2 float_to_fp6(float x) {
    if (isnan(x)) {
        // Represent NaN as max exponent, non-zero mantissa
        return nv_fp6_e3m2( (uint8_t)((FP6_EXP_MAX<<2) | 0x1) );
    }
    if (isinf(x)) {
        return nv_fp6_e3m2( (uint8_t)(( (signbit(x)?1:0 )<<5) | (FP6_EXP_MAX<<2)) );
    }
    if (x == 0.0f) {
        return nv_fp6_e3m2(0);
    }

    int sign = x < 0.0f; if (sign) x = -x;

    int exp;
    float mant = frexpf(x, &exp); // x = mant * 2^(exp), mant in [0.5,1)
    // Convert to unbiased exponent E = exp-1 (since mant in [0.5,1) ~ 1.m form). For IEEE representation conceptually.
    int E = exp - 1;

    int encoded_exp = E + FP6_EXP_BIAS;

    if (encoded_exp <= 0) {
        // Subnormal or underflow to zero
        if (encoded_exp < -FP6_MAN_BITS) {
            return nv_fp6_e3m2( (uint8_t)(sign<<5) );
        }
        // create subnormal: shift mantissa accordingly
        // scale original x to subnormal mantissa: m = x / 2^(1 - bias - man_bits)
        float scaled = ldexpf(mant, E - ( -FP6_EXP_BIAS + 1)); // approximate
        int man = (int)lrintf(scaled * (1<<FP6_MAN_BITS));
        if (man == 0) return nv_fp6_e3m2( (uint8_t)(sign<<5) );
        if (man > FP6_MAN_MAX) man = FP6_MAN_MAX;
        uint8_t bits = (uint8_t)((sign<<5) | man);
        return nv_fp6_e3m2(bits);
    }
    if (encoded_exp >= FP6_EXP_MAX) {
        // Overflow -> Inf
        return nv_fp6_e3m2( (uint8_t)((sign<<5) | (FP6_EXP_MAX<<2)) );
    }

    // Remove implicit leading 1: mant in [0.5,1) -> convert to [1,2) by *2 then subtract 1
    float frac = (mant * 2.0f) - 1.0f; // in [0,1)
    int man = (int)lrintf(frac * (1<<FP6_MAN_BITS));
    if (man > FP6_MAN_MAX) {
        // Rounding caused mantissa overflow; adjust exponent
        man = 0;
        encoded_exp += 1;
        if (encoded_exp >= FP6_EXP_MAX) {
            return nv_fp6_e3m2( (uint8_t)((sign<<5) | (FP6_EXP_MAX<<2)) );
        }
    }

    uint8_t bits = (uint8_t)((sign<<5) | (encoded_exp<<2) | man);
    return nv_fp6_e3m2(bits);
}

__host__ __device__ inline float fp6_to_float(nv_fp6_e3m2 v) {
    uint8_t bits = v.data & 0x3F;
    int sign = (bits>>5) & 0x1;
    int expField = (bits>>2) & 0x7;
    int man = bits & 0x3;

    if (expField == FP6_EXP_MAX) {
        if (man == 0) return sign ? -INFINITY : INFINITY;
        return NAN; // NaN
    }
    if (expField == 0) {
        if (man == 0) return sign ? -0.0f : 0.0f;
        // subnormal: exponent = 1 - bias
        float frac = (float)man / (float)(1<<FP6_MAN_BITS);
        float val = ldexpf(frac, 1 - FP6_EXP_BIAS - FP6_MAN_BITS);
        return sign ? -val : val;
    }
    int E = expField - FP6_EXP_BIAS; // unbiased exponent
    float frac = 1.0f + (float)man / (float)(1<<FP6_MAN_BITS);
    float val = ldexpf(frac, E);
    return sign ? -val : val;
}

// Device kernels
__global__ void k_fp32_to_fp6(const float* __restrict__ in, nv_fp6_e3m2* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = float_to_fp6(in[i]);
    }
}

__global__ void k_fp6_to_fp32(const nv_fp6_e3m2* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = fp6_to_float(in[i]);
    }
}

void convert_fp32_to_fp6(const float* d_in, nv_fp6_e3m2* d_out, size_t n, cudaStream_t stream) {
    int block = 256;
    auto grid = makeGrid(n, block);
    k_fp32_to_fp6<<<grid, block, 0, stream>>>(d_in, d_out, n);
}

void convert_fp6_to_fp32(const nv_fp6_e3m2* d_in, float* d_out, size_t n, cudaStream_t stream) {
    int block = 256;
    auto grid = makeGrid(n, block);
    k_fp6_to_fp32<<<grid, block, 0, stream>>>(d_in, d_out, n);
}
