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
    printf("  conversion_id: 0 for float to fp6_e2m3\n");
    printf("  conversion_id: 1 for float to fp6_e3m2\n");
    printf("  conversion_id: 2 for float to fp8_e4m3\n");
    printf("  conversion_id: 3 for float to fp8_e5m2\n");

}

int main(int argc, char* argv[]) 
{
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    float testVal = atof(argv[1]);
    int conversion_id = atoi(argv[2]);
    
    if (conversion_id < 0 || conversion_id > 3) {
        printf("Error: conversion_id must be 0 (fp6_e2m3), 1 (fp6_e3m2), 2 (fp8_e4m3), or 3 (fp8_e5m2)\n");
        print_usage(argv[0]);
        return 1;
    }

    const char* format_names[] = {"fp6_e2m3", "fp6_e3m2", "fp8_e4m3", "fp8_e5m2"};
    printf("%s conversion test testVal %f\n", format_names[conversion_id], testVal);
    
    unsigned char testOut = 0;
    float float32Out = 0.0f;

    switch (conversion_id) {
        case 0: // fp6_e2m3
            ConvertKernel_fp32_to_fp6_e2m3_CUDA(&testVal, &testOut);
            printf("fp32 %f -> fp6_e2m3 0x%02x (%d)\n", testVal, testOut, testOut);
            ConvertKernel_fp6_e2m3_to_fp32_CUDA(&testOut, &float32Out);
            printf("fp6_e2m3 0x%02x -> fp32 %f\n", testOut, float32Out);
            break;
        case 1: // fp6_e3m2
            ConvertKernel_fp32_to_fp6_e3m2_CUDA(&testVal, &testOut);
            printf("fp32 %f -> fp6_e3m2 0x%02x (%d)\n", testVal, testOut, testOut);
            ConvertKernel_fp6_e3m2_to_fp32_CUDA(&testOut, &float32Out);
            printf("fp6_e3m2 0x%02x -> fp32 %f\n", testOut, float32Out);
            break;
        case 2: // fp8_e4m3
            ConvertKernel_fp32_to_fp8_e4m3_CUDA(&testVal, &testOut);
            printf("fp32 %f -> fp8_e4m3 0x%02x (%d)\n", testVal, testOut, testOut);
            ConvertKernel_fp8_e4m3_to_fp32_CUDA(&testOut, &float32Out);
            printf("fp8_e4m3 0x%02x -> fp32 %f\n", testOut, float32Out);
            break;
        case 3: // fp8_e5m2
            ConvertKernel_fp32_to_fp8_e5m2_CUDA(&testVal, &testOut);
            printf("fp32 %f -> fp8_e5m2 0x%02x (%d)\n", testVal, testOut, testOut);
            ConvertKernel_fp8_e5m2_to_fp32_CUDA(&testOut, &float32Out);
            printf("fp8_e5m2 0x%02x -> fp32 %f\n", testOut, float32Out);
            break;
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