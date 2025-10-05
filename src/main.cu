#include "fp6.cuh"
#include "fp6_cuda.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void print_usage(const char* prog_name) {
    printf("Usage: %s <direction> <value> <format_id>\n", prog_name);
    printf("  direction: 'compress' for float32 to fp conversion, 'expand' for fp to float32 conversion\n");
    printf("  value: \n");
    printf("    - for compress: floating point value (e.g., 3.14159)\n");
    printf("    - for expand: hex value with 0x prefix (e.g., 0x42) or decimal integer\n");
    printf("  format_id: 0 for fp6_e2m3, 1 for fp6_e3m2, 2 for fp8_e4m3, 3 for fp8_e5m2\n");
    printf("\nExamples:\n");
    printf("  %s compress 3.14159 1    # Convert float 3.14159 to fp6_e3m2\n", prog_name);
    printf("  %s expand 0x42 1         # Convert hex 0x42 from fp6_e3m2 to float\n", prog_name);
    printf("  %s expand 66 1           # Convert decimal 66 from fp6_e3m2 to float\n", prog_name);
}

int main(int argc, char* argv[]) 
{
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* direction = argv[1];
    const char* value_str = argv[2];
    int format_id = atoi(argv[3]);
    
    if (format_id < 0 || format_id > 3) {
        printf("Error: format_id must be 0 (fp6_e2m3), 1 (fp6_e3m2), 2 (fp8_e4m3), or 3 (fp8_e5m2)\n");
        print_usage(argv[0]);
        return 1;
    }

    bool is_compress = false; // compress: float to fp
    bool is_expand = false;   // expand: fp to float
    
    if (strcmp(direction, "compress") == 0) {
        is_compress = true;
    } else if (strcmp(direction, "expand") == 0) {
        is_expand = true;
    } else {
        printf("Error: direction must be 'compress' or 'expand'\n");
        print_usage(argv[0]);
        return 1;
    }

    const char* format_names[] = {"fp6_e2m3", "fp6_e3m2", "fp8_e4m3", "fp8_e5m2"};
    
    float input_float = 0.0f;
    unsigned char input_fp = 0;
    unsigned char output_fp = 0;
    float output_float = 0.0f;
    
    if (is_compress) {
        // Convert string to float for compression (float -> fp)
        input_float = atof(value_str);
        printf("%s compression: float %f\n", format_names[format_id], input_float);
        
        // Perform forward conversion based on format
        switch (format_id) {
            case 0: // fp6_e2m3
                ConvertKernel_fp32_to_fp6_e2m3_CUDA(&input_float, &output_fp);
                printf("fp32 %f -> fp6_e2m3 0x%02x (%d)\n", input_float, output_fp, output_fp);
                break;
            case 1: // fp6_e3m2
                ConvertKernel_fp32_to_fp6_e3m2_CUDA(&input_float, &output_fp);
                printf("fp32 %f -> fp6_e3m2 0x%02x (%d)\n", input_float, output_fp, output_fp);
                break;
            case 2: // fp8_e4m3
                ConvertKernel_fp32_to_fp8_e4m3_CUDA(&input_float, &output_fp);
                printf("fp32 %f -> fp8_e4m3 0x%02x (%d)\n", input_float, output_fp, output_fp);
                break;
            case 3: // fp8_e5m2
                ConvertKernel_fp32_to_fp8_e5m2_CUDA(&input_float, &output_fp);
                printf("fp32 %f -> fp8_e5m2 0x%02x (%d)\n", input_float, output_fp, output_fp);
                break;
        }
    } else if (is_expand) {
        // Parse input as hex (0x prefix) or decimal integer
        if (strncmp(value_str, "0x", 2) == 0 || strncmp(value_str, "0X", 2) == 0) {
            input_fp = (unsigned char)strtol(value_str, NULL, 16);
        } else {
            input_fp = (unsigned char)atoi(value_str);
        }
        
        printf("%s expansion: 0x%02x (%d)\n", format_names[format_id], input_fp, input_fp);
        
        // Perform reverse conversion based on format
        switch (format_id) {
            case 0: // fp6_e2m3
                ConvertKernel_fp6_e2m3_to_fp32_CUDA(&input_fp, &output_float);
                printf("fp6_e2m3 0x%02x -> fp32 %f\n", input_fp, output_float);
                break;
            case 1: // fp6_e3m2
                ConvertKernel_fp6_e3m2_to_fp32_CUDA(&input_fp, &output_float);
                printf("fp6_e3m2 0x%02x -> fp32 %f\n", input_fp, output_float);
                break;
            case 2: // fp8_e4m3
                ConvertKernel_fp8_e4m3_to_fp32_CUDA(&input_fp, &output_float);
                printf("fp8_e4m3 0x%02x -> fp32 %f\n", input_fp, output_float);
                break;
            case 3: // fp8_e5m2
                ConvertKernel_fp8_e5m2_to_fp32_CUDA(&input_fp, &output_float);
                printf("fp8_e5m2 0x%02x -> fp32 %f\n", input_fp, output_float);
                break;
        }
    }

    // Optional: Show roundtrip conversion for verification
    if (is_compress) {
        // Do a roundtrip to show accuracy
        switch (format_id) {
            case 0:
                ConvertKernel_fp6_e2m3_to_fp32_CUDA(&output_fp, &output_float);
                break;
            case 1:
                ConvertKernel_fp6_e3m2_to_fp32_CUDA(&output_fp, &output_float);
                break;
            case 2:
                ConvertKernel_fp8_e4m3_to_fp32_CUDA(&output_fp, &output_float);
                break;
            case 3:
                ConvertKernel_fp8_e5m2_to_fp32_CUDA(&output_fp, &output_float);
                break;
        }
        float error = fabsf(input_float - output_float);
        printf("Roundtrip: %f -> 0x%02x -> %f (absolute error: %e)\n", input_float, output_fp, output_float, error);
    }

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