#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Error checking function
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(-1);
    }
}

int main() {
    int N = 1000000; // Size of vectors
    size_t size = N * sizeof(int);
    
    // Allocate host memory
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        cout << "Host memory allocation failed!" << endl;
        return -1;
    }
    
    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    checkCudaError(err, "cudaMalloc for d_A");
    err = cudaMalloc((void**)&d_B, size);
    checkCudaError(err, "cudaMalloc for d_B");
    err = cudaMalloc((void**)&d_C, size);
    checkCudaError(err, "cudaMalloc for d_C");
    
    // Copy data to device
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy A to device");
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    checkCudaError(err, "cudaMemcpy B to device");
    
    // Define block and grid size
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");

    // Synchronize to ensure completion of kernel
    err = cudaDeviceSynchronize();
    checkCudaError(err, "Device synchronization failed");

    // Copy result back to host
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "cudaMemcpy C to host");
    
    // Print sample result
    cout << "Sample result: " << h_C[0] << ", " << h_C[N-1] << endl;
    
    // Free memory
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}

