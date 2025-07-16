#include <cuda_runtime.h>
#include "sq_matmul.cuh"
#include <iostream>

__global__
void sq_matmul_uncoalesced(const float* A, const float* B, float* C, int N){
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N && j < N){
        float value = 0.0f;
        for (int k = 0; k < N; ++k){
            value += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = value;
    }
    return;
}

__global__
void sq_matmul_coalesced(const float* A, const float* B, float* C, int N){
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


void launch_sq_matmul(const float* A, const float* B, float* C, int N, const std::string& version){
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    if (version == "uncoalesced"){
        sq_matmul_uncoalesced<<<grid, block>>>(A, B, C, N);
    }else if (version == "coalesced"){
        sq_matmul_coalesced<<<grid, block>>>(A, B, C, N);
    }else{
        std::cout << "please select type of operation" << std::endl;
    }
}
