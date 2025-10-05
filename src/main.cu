#include "fp6.cuh"
#include "fp6_cuda.cuh"
#include <cassert>
#include <cstdio>
#include <vector>







int main() 
{
    float testVal = 7.4f;
    printf("FP6<e2m3> conversion test testVal %f\n", testVal);
    unsigned char testOut = 0;

    ConvertKernel_fp32_to_fp6_e2m3_CUDA(&testVal, &(testOut), 1);

    printf("output one way = %d\n", testOut);
    
    float float32Out = 0.0f;

    ConvertKernel_fp6_e2m3_to_fp32_CUDA(&(testOut), &float32Out, 1);


    printf("output other way = %f\n", float32Out); 

    const size_t N = 8;
    float h_vals[N] = {0.f, 1.f, -1.f, 0.5f, 3.2f, -2.75f, 10.f, 1e-2f};

    void* pfp6OutputBuffer = malloc(sizeof(nv_fp6_e3m2) * N);
    void* pfp32OutputBuffer = malloc(sizeof(float) * N);

    ConvertKernel_fp32_to_fp6(h_vals, (nv_fp6_e3m2*)pfp6OutputBuffer, N);
    ConvertKernel_fp6_to_fp32((nv_fp6_e3m2*)pfp6OutputBuffer, (float*)pfp32OutputBuffer, N);

    free(pfp6OutputBuffer);
    free(pfp32OutputBuffer);

    return 0;
}