// nvcc l8_shared_mem_matmul.cu -o out && ncu --set full -f -o report.ncu ./out
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

const int matrix_size = 128; //square matrix

//from NVIDIA, for error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

#define flatIdx(row, col, width) ((row) * (width) + (col)) //row maj

template <typename T>
__global__ void smem_matmul(const T *A, const T *B, T *C, const int arr_size){
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    constexpr int TILE_SIZE = 32;
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    T _temp = T(0);
    int steps = (arr_size+TILE_SIZE-1)/TILE_SIZE;

    for (int t=0; t<steps; ++t){
        bool Acheck = (row<arr_size) && (( t*TILE_SIZE+threadIdx.x)<arr_size);
        bool Bcheck = (col<arr_size) && (( t*TILE_SIZE+threadIdx.y)<arr_size);
        As[threadIdx.y][threadIdx.x] = Acheck ? A[flatIdx(row, t*TILE_SIZE+threadIdx.x, arr_size)] : T(0);
        Bs[threadIdx.y][threadIdx.x] = Bcheck ? B[flatIdx(t*TILE_SIZE+threadIdx.y, col, arr_size)] : T(0);
        __syncthreads();

        for (int k=0; k<TILE_SIZE; ++k){
            _temp += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    if ((row<arr_size)&&(col<arr_size)){
    C[flatIdx(row,col,arr_size)] = _temp;};
   
}


int main(){

    cudaEvent_t start, stop; //for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *h_check;
    float temp;
    h_A = new float[matrix_size*matrix_size];
    h_B = new float[matrix_size*matrix_size];
    h_C = new float[matrix_size*matrix_size];
    h_check = new float[matrix_size*matrix_size];
    for (int i = 0; i < matrix_size*matrix_size; i++){
        h_A[i] = i % 100;
        h_B[i] = i % 15;
    }
    for (int i=0; i<matrix_size; i++){
        for (int j=0; j<matrix_size; j++){
            temp = 0.0f;
            for (int k=0; k<matrix_size; k++){ 
                temp += h_A[i*matrix_size+k] * h_B[k*matrix_size+j];
            }
            h_check[i*matrix_size+j] = temp;
        }
    }

    cudaMalloc(&d_A, matrix_size*matrix_size*sizeof(float));
    cudaMalloc(&d_B, matrix_size*matrix_size*sizeof(float));
    cudaMalloc(&d_C, matrix_size*matrix_size*sizeof(float));

    cudaMemcpy(d_A, h_A, matrix_size*matrix_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size*matrix_size*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("allocated and pushed to device");

    const int thread_per_block = 32;
    const int block_per_grid = (matrix_size+thread_per_block-1)/(thread_per_block);
    dim3 block_dim(thread_per_block, thread_per_block,1);
    dim3 grid_dim(block_per_grid, block_per_grid, 1);

    cudaEventRecord(start);
    smem_matmul<float><<<grid_dim, block_dim>>>(d_A, d_B, d_C, matrix_size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaCheckErrors("matmul kernel completed");
    cudaMemcpy(h_C, d_C, matrix_size*matrix_size*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("result is back on host");

    float runtime = 0;
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Kernel execution time: %f ms\n", runtime);

    //verify the result
    for (int i=0; i<matrix_size*matrix_size; i++){
           if(h_check[i] != h_C[i]){
            printf("result at index %d does not match, expected %f, got %f\n", i, h_check[i], h_C[i]);
            return EXIT_FAILURE;
           }
        }
    printf("Passed! \n");

    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_check;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return EXIT_SUCCESS;
}