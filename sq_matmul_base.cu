#include <cuda_runtime.h>
#include "sq_matmul_base.cuh"

__global__
void sq_matmul_base(float* A, float* B, float* C, int N){
    // int j = blockDim.y * blockIdx.y + threadIdx.y;
    // int i = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N && j < N){
        float value = 0.0f;
        for (int k = 0; k < N; ++k){
            value += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = value;
    }
    return;
}

void launch_sq_matmul(float* A, float* B, float* C, int N){
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    sq_matmul_base<<<grid, block>>>(A, B, C, N);
}
