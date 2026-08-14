// ROCmFP4 type definitions for CUDA operations
#ifndef QK_ROCMFP4
#define QK_ROCMFP4 32
#endif
#ifndef QR_ROCMFP4
#define QR_ROCMFP4 2
#endif
typedef struct { uint8_t qs[QK_ROCMFP4/2]; uint8_t e[2]; } block_rocmfp4;
typedef struct { uint8_t qs[QK_ROCMFP4/2]; uint8_t e; } block_rocmfp4_fast;


#include "common.cuh"

#define CUDA_CPY_BLOCK_SIZE 64

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1);

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
