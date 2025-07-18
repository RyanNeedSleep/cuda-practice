#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include "sq_matmul.cuh"

using namespace std;

// SELECT TYPE OF IMPLEMENTATION
const string TYPE = "shared";
int ARRAY_SIZE = 1024;

int main(void){
   float* A, *B, *C;
   size_t bytes = ARRAY_SIZE * ARRAY_SIZE * sizeof(float);
   cudaMallocManaged(&A, bytes);
   cudaMallocManaged(&B, bytes);
   cudaMallocManaged(&C, bytes);

   for (int i = 0; i < ARRAY_SIZE; ++i){
       for (int j = 0; j < ARRAY_SIZE; ++j){
           A[i * ARRAY_SIZE + j] = static_cast<float>(i + j);
           B[i * ARRAY_SIZE + j] = static_cast<float>(i + j) * 0.5f ;
       }
   }

   launch_sq_matmul(A, B, C, ARRAY_SIZE, TYPE);

   cudaDeviceSynchronize();

   double max_abs_err = 0.0;
   for (int i = 0; i < ARRAY_SIZE; ++i){
        for (int j = 0; j < ARRAY_SIZE; ++j){
            float ref = 0.f;
            for (int k = 0; k < ARRAY_SIZE; ++k){
                ref += A[i * ARRAY_SIZE + k] * B[k * ARRAY_SIZE + j];
            }
            float gpu = C[i * ARRAY_SIZE + j];
            double err = std::fabs(static_cast<double>(gpu) - ref);
            if (err > max_abs_err) max_abs_err = err;
        }
    }

    std::cout << "Max ABS Error: " << max_abs_err << "\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}
