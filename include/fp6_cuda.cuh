 __host__ __device__  void ConvertKernel_fp32_to_fp6_e2m3_CUDA(const float *pIn, unsigned char *pOut, size_t N) ;

 __host__ __device__  void ConvertKernel_fp6_e2m3_to_fp32_CUDA(const unsigned char *pIn, float *pOut, size_t N) ;

 __host__ __device__  void ConvertKernel_fp32_to_fp6_e3m2_CUDA(const float *pIn, unsigned char *pOut, size_t N) ;

 __host__ __device__  void ConvertKernel_fp6_e3m2_to_fp32_CUDA(const unsigned char *pIn, float *pOut, size_t N) ;