#include "fp6.cuh"
#include <vector>
#include <iostream>
#include <random>

int main() {
    const size_t N = 32;
    std::vector<float> h_in(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);
    for (auto &v : h_in) v = dist(rng);

    float* d_in; nv_fp6_e3m2* d_q; float* d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_q, N*sizeof(nv_fp6_e3m2));
    cudaMalloc(&d_out, N*sizeof(float));

    cudaMemcpy(d_in, h_in.data(), N*sizeof(float), cudaMemcpyHostToDevice);

    convert_fp32_to_fp6(d_in, d_q, N);
    convert_fp6_to_fp32(d_q, d_out, N);

    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_out, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for (size_t i=0;i<N;i++) {
        std::cout << i << ": " << h_in[i] << " -> " << h_out[i] << " (abs err=" << fabs(h_in[i]-h_out[i]) << ")\n";
    }

    cudaFree(d_in); cudaFree(d_q); cudaFree(d_out);
    return 0;
}
