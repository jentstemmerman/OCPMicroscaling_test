#include "fp6_cuda.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Helper function to convert single fp value to float based on format
float convert_fp_to_float(unsigned char fp_value, int format_id) {
    float result = 0.0f;
    switch (format_id) {
        case 0:
            ConvertKernel_fp6_e2m3_to_fp32_CUDA(&fp_value, &result);
            break;
        case 1:
            ConvertKernel_fp6_e3m2_to_fp32_CUDA(&fp_value, &result);
            break;
        case 2:
            ConvertKernel_fp8_e4m3_to_fp32_CUDA(&fp_value, &result);
            break;
        case 3:
            ConvertKernel_fp8_e5m2_to_fp32_CUDA(&fp_value, &result);
            break;
    }
    return result;
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [--quiet] <direction> <value> <format_id>\n", prog_name);
    printf("  --quiet: optional flag to output only the result value (for use with other tools)\n");
    printf("  direction: 'compress' for float32 to fp conversion, 'expand' for fp to float32 conversion\n");
    printf("  value: \n");
    printf("    - for compress: floating point value (e.g., 3.14159)\n");
    printf("    - for expand: single hex/decimal value OR range (e.g., 0x42, 66, or 0x10-0x20)\n");
    printf("  format_id: 0 for fp6_e2m3, 1 for fp6_e3m2, 2 for fp8_e4m3, 3 for fp8_e5m2\n");
    printf("\nExamples:\n");
    printf("  %s compress 3.14159 1        # Convert float 3.14159 to fp6_e3m2\n", prog_name);
    printf("  %s expand 0x42 1             # Convert hex 0x42 from fp6_e3m2 to float\n", prog_name);
    printf("  %s expand 0x10-0x20 1        # Convert range 0x10 to 0x20 from fp6_e3m2 to float\n", prog_name);
    printf("  %s --quiet compress 3.14 1   # Output only the hex result\n", prog_name);
    printf("  %s --quiet expand 0x42 1     # Output only the float result\n", prog_name);
}

int main(int argc, char* argv[]) 
{
    bool quiet_mode = false;
    int arg_offset = 0;
    
    // Check for --quiet flag
    if (argc > 1 && strcmp(argv[1], "--quiet") == 0) {
        quiet_mode = true;
        arg_offset = 1;
    }
    
    if (argc != 4 + arg_offset) {
        print_usage(argv[0]);
        return 1;
    }

    const char* direction = argv[1 + arg_offset];
    const char* value_str = argv[2 + arg_offset];
    int format_id = atoi(argv[3 + arg_offset]);
    
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
        if (!quiet_mode) {
            printf("%s compression ", format_names[format_id]);
        }
        
        // Perform forward conversion based on format
        switch (format_id) {
            case 0: // fp6_e2m3
                ConvertKernel_fp32_to_fp6_e2m3_CUDA(&input_float, &output_fp);
                if (quiet_mode) {
                    printf("0x%02x\n", output_fp);
                } else {
                    printf("fp32 %f -> fp6_e2m3 0x%02x (%d)\n", input_float, output_fp, output_fp);
                }
                break;
            case 1: // fp6_e3m2
                ConvertKernel_fp32_to_fp6_e3m2_CUDA(&input_float, &output_fp);
                if (quiet_mode) {
                    printf("0x%02x\n", output_fp);
                } else {
                    printf("fp32 %f -> fp6_e3m2 0x%02x (%d)\n", input_float, output_fp, output_fp);
                }
                break;
            case 2: // fp8_e4m3
                ConvertKernel_fp32_to_fp8_e4m3_CUDA(&input_float, &output_fp);
                if (quiet_mode) {
                    printf("0x%02x\n", output_fp);
                } else {
                    printf("fp32 %f -> fp8_e4m3 0x%02x (%d)\n", input_float, output_fp, output_fp);
                }
                break;
            case 3: // fp8_e5m2
                ConvertKernel_fp32_to_fp8_e5m2_CUDA(&input_float, &output_fp);
                if (quiet_mode) {
                    printf("0x%02x\n", output_fp);
                } else {
                    printf("fp32 %f -> fp8_e5m2 0x%02x (%d)\n", input_float, output_fp, output_fp);
                }
                break;
        }
    } else if (is_expand) {
        // Check if value_str contains a range (e.g., "0x10-0x20")
        // Make a mutable copy of value_str to work with
        char value_copy[256];
        strncpy(value_copy, value_str, sizeof(value_copy) - 1);
        value_copy[sizeof(value_copy) - 1] = '\0';
        
        char* dash_pos = strchr(value_copy, '-');
        
        if (dash_pos != NULL) {
            // Handle range expansion
            *dash_pos = '\0'; // Split the string
            char* start_str = value_copy;
            char* end_str = dash_pos + 1;
            
            unsigned char start_val, end_val;
            
            // Parse start value
            if (strncmp(start_str, "0x", 2) == 0 || strncmp(start_str, "0X", 2) == 0) {
                start_val = (unsigned char)strtol(start_str, NULL, 16);
            } else {
                start_val = (unsigned char)atoi(start_str);
            }
            
            // Parse end value
            if (strncmp(end_str, "0x", 2) == 0 || strncmp(end_str, "0X", 2) == 0) {
                end_val = (unsigned char)strtol(end_str, NULL, 16);
            } else {
                end_val = (unsigned char)atoi(end_str);
            }
            
            if (!quiet_mode) {
                printf("%s expansion range: 0x%02x to 0x%02x\n", format_names[format_id], start_val, end_val);
            }
            
            // Loop through the range
            for (unsigned char val = start_val; val <= end_val; val++) {
                float result = convert_fp_to_float(val, format_id);
                
                if (quiet_mode) {
                    printf("%f\n", result);
                } else {
                    printf("%s 0x%02x -> fp32 %f\n", format_names[format_id], val, result);
                }
                
                // Prevent infinite loop if end_val is 255 (max unsigned char)
                if (val == end_val) break;
            }
        } else {
            // Handle single value expansion (original behavior)
            if (strncmp(value_str, "0x", 2) == 0 || strncmp(value_str, "0X", 2) == 0) {
                input_fp = (unsigned char)strtol(value_str, NULL, 16);
            } else {
                input_fp = (unsigned char)atoi(value_str);
            }
            
            if (!quiet_mode) {
                printf("%s expansion: 0x%02x (%d)\n", format_names[format_id], input_fp, input_fp);
            }
            
            output_float = convert_fp_to_float(input_fp, format_id);
            
            if (quiet_mode) {
                printf("%f\n", output_float);
            } else {
                printf("%s 0x%02x -> fp32 %f\n", format_names[format_id], input_fp, output_float);
            }
        }
    }

    // Optional: Show roundtrip conversion for verification (only in verbose mode)
    if (is_compress && !quiet_mode) {
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
    return 0;
}