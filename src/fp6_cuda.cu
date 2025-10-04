#include "fp6.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_fp6.h>



__global__ void ConvertKernel_fp32_to_fp6_CUDA(const float *pIn, nv_fp6_e3m2 *pOut, size_t N) 
{
    float x = 10.1f;


    
    // Convert float -> FP6
    struct __nv_fp6_e2m3 a = __nv_fp6_e2m3(x); 
   
    // Convert back FP6 -> float
    //float y = __nv_cvt_fp6_e2m3_to_fp32(a);

}