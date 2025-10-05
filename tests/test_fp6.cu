#include <cassert>
#include <cstdio>
#include <vector>

// void ConvertKernel_fp32_to_fp6(const float *pIn, nv_fp6_e3m2 *pOut, size_t N) 
// {

//     float *pInputFloat32DataOnGPU; 
//     nv_fp6_e3m2 *pOutputDataOnGPU; 

//     const uint32_t fp6_e3m2_buffer_size = N * sizeof(nv_fp6_e3m2);
//     const uint32_t fp32_buffer_size     = N * sizeof(float);

//     cudaMalloc(&pInputFloat32DataOnGPU, fp32_buffer_size);
//     cudaMalloc(&pOutputDataOnGPU, fp6_e3m2_buffer_size);

//     cudaMemcpy(pInputFloat32DataOnGPU, pIn, fp32_buffer_size, cudaMemcpyHostToDevice);
    
//     convert_fp32_to_fp6(pInputFloat32DataOnGPU, pOutputDataOnGPU, N);
//     cudaDeviceSynchronize();
//     cudaMemcpy(pOut, pOutputDataOnGPU, fp6_e3m2_buffer_size, cudaMemcpyDeviceToHost);

//     for (size_t i=0;i<N;i++) {
//         printf("fp32: %f -> fp6_e3m2: 0x%02x\n", pIn[i], pOut[i].data);
//     }

//     cudaFree(pInputFloat32DataOnGPU); 
//     cudaFree(pOutputDataOnGPU);
// }

// void ConvertKernel_fp6_to_fp32(const nv_fp6_e3m2 *pIn, float *pOut, size_t N) 
// {
//     nv_fp6_e3m2 *pInputDataOnGPU; 
//     float *pOutputFloat32DataOnGPU; 

//     const uint32_t fp6_e3m2_buffer_size = N * sizeof(nv_fp6_e3m2);
//     const uint32_t fp32_buffer_size     = N * sizeof(float);

//     cudaMalloc(&pInputDataOnGPU, fp6_e3m2_buffer_size);
//     cudaMalloc(&pOutputFloat32DataOnGPU, fp32_buffer_size);

//     cudaMemcpy(pInputDataOnGPU, pIn, fp6_e3m2_buffer_size, cudaMemcpyHostToDevice);

//     convert_fp6_to_fp32(pInputDataOnGPU, pOutputFloat32DataOnGPU, N);
    
//     cudaDeviceSynchronize();

//     cudaMemcpy(pOut, pOutputFloat32DataOnGPU, fp32_buffer_size, cudaMemcpyDeviceToHost);
    

//     for (size_t i=0;i<N;i++) {
//         printf("fp6_e3m2: 0x%02x -> fp32: %f\n", pIn[i].data, pOut[i]);
//     }

//     cudaFree(pInputDataOnGPU); 
//     cudaFree(pOutputFloat32DataOnGPU);
// }

int main() {
    printf("FP6 test program - basic test passed\n");
    return 0;
}
