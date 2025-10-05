#include "fp6.cuh"
#include "fp6_cuda.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void print_usage(const char* prog_name) {
    printf("Usage: %s <testVal> <conversion_id>\n", prog_name);
    printf("  testVal: floating point value to convert\n");
    printf("  conversion_id: 0 for e2m3\n");
    printf("  conversion_id: 1 for e3m2\n");


}

int main(int argc, char* argv[]) 
{
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    float testVal = atof(argv[1]);
    int conversion_id = atoi(argv[2]);
    
    if (conversion_id != 0 && conversion_id != 1) {
        printf("Error: conversion_id must be 0 (e2m3) or 1 (e3m2)\n");
        print_usage(argv[0]);
        return 1;
    }

    const char* format_name = (conversion_id == 0) ? "e2m3" : "e3m2";
    printf("FP6<%s> conversion test testVal %f\n", format_name, testVal);
    
    unsigned char testOut = 0;
    float float32Out = 0.0f;

    if (conversion_id == 0) {
        // e2m3 conversion
        ConvertKernel_fp32_to_fp6_e2m3_CUDA(&testVal, &testOut, 1);
        printf("fp32 %f -> fp6_e2m3 0x%02x (%d)\n", testVal, testOut, testOut);
        
        ConvertKernel_fp6_e2m3_to_fp32_CUDA(&testOut, &float32Out, 1);
        printf("fp6_e2m3 0x%02x -> fp32 %f\n", testOut, float32Out);
    } else {
        // e3m2 conversion
        ConvertKernel_fp32_to_fp6_e3m2_CUDA(&testVal, &testOut, 1);
        printf("fp32 %f -> fp6_e3m2 0x%02x (%d)\n", testVal, testOut, testOut);
        
        ConvertKernel_fp6_e3m2_to_fp32_CUDA(&testOut, &float32Out, 1);
        printf("fp6_e3m2 0x%02x -> fp32 %f\n", testOut, float32Out);
    }
    
    float error = fabsf(testVal - float32Out);
    printf("Roundtrip error: %f -> %f (absolute error: %e)\n", testVal, float32Out, error);

    // Optional: run batch conversion test with predefined values
    printf("\nBatch conversion test with 8 predefined values:\n");
    const size_t N = 8;
    float h_vals[N] = {0.f, 1.f, -1.f, 0.5f, 3.2f, -2.75f, 10.f, 1e-2f};

    void* pfp6OutputBuffer = malloc(sizeof(nv_fp6_e3m2) * N);
    void* pfp32OutputBuffer = malloc(sizeof(float) * N);

    // Use the GPU-based batch conversion functions from fp6.cu
    convert_fp32_to_fp6(h_vals, (nv_fp6_e3m2*)pfp6OutputBuffer, N);
    convert_fp6_to_fp32((nv_fp6_e3m2*)pfp6OutputBuffer, (float*)pfp32OutputBuffer, N);

    free(pfp6OutputBuffer);
    free(pfp32OutputBuffer);

    return 0;
}