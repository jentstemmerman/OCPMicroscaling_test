#include "fp6.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_fp6.h>

__host__ __device__ float half_raw_to_float(__half_raw h_raw) {
    __half h = *reinterpret_cast<__half*>(&h_raw); // reinterpret raw bits as __half
    return __half2float(h); // convert to float32
}

 __host__ __device__  void ConvertKernel_fp32_to_fp6_e2m3_CUDA(const float *pIn, unsigned char *pOut, size_t N) 
{
    float x = *pIn;
    struct __nv_fp6_e2m3 a = __nv_fp6_e2m3(x); 
    
    pOut[0] = (uint8_t)a.__x; 
}

 __host__ __device__  void ConvertKernel_fp6_e2m3_to_fp32_CUDA(const unsigned char *pIn, float *pOut, size_t N) 
{
    __nv_fp6_storage_t x = *reinterpret_cast<const __nv_fp6_storage_t*>(pIn);
    __nv_fp6_interpretation_t  interpretation = __NV_E2M3;
    struct __nv_fp6_e2m3 a = __nv_fp6_e2m3(x);

    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(x, interpretation);
    *pOut = half_raw_to_float(half_raw);
}

 __host__ __device__  void ConvertKernel_fp32_to_fp6_e3m2_CUDA(const float *pIn, unsigned char *pOut, size_t N) 
{
    float x = *pIn;
    struct __nv_fp6_e3m2 a = __nv_fp6_e3m2(x);

    pOut[0] = (uint8_t)a.__x;
}

 __host__ __device__  void ConvertKernel_fp6_e3m2_to_fp32_CUDA(const unsigned char *pIn, float *pOut, size_t N) 
{
    __nv_fp6_storage_t x = *reinterpret_cast<const __nv_fp6_storage_t*>(pIn);
    __nv_fp6_interpretation_t  interpretation = __NV_E3M2;
   
    
    struct __nv_fp6_e3m2 a = __nv_fp6_e3m2(x);

    __half_raw half_raw = __nv_cvt_fp6_to_halfraw(x, interpretation);
    *pOut = half_raw_to_float(half_raw);
}


 __host__ __device__  void ConvertKernel_int_to_scale_CUDA(const float *pIn, __nv_fp8x2_e8m0 *pOut, size_t N) 
{
    float2 a = {*pIn, *pIn}; 

    struct __nv_fp8x2_e8m0 b = __nv_fp8x2_e8m0(a);
    
    *pOut = b;
}

