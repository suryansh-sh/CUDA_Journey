//nvcc -lineinfo -o out dot_prod.cu && ncu  --set full > K.log ./out
//ncu  --set full --export dot_prod.ncu-rep -f ./out
#include <iostream>
#include <cmath>
#include <numeric>
#include <assert.h> 

//#define K1
//#define K2
//#define K3
#define K4


#define N (1<<20)

template <typename T>
__global__ void atomic_dp(T* a, T* b, T* res, int n){ //K1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    atomicAdd(res, a[idx]*b[idx]);
}

template <typename T>
__global__ void sh_mem_atomic(T *a, T* b, T *res, int n){ //K2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return; //this may create problems with sync threads
    __shared__ T shmem[1024]; //using 1024 thread blocks
    shmem[threadIdx.x] = a[idx]*b[idx];
    __syncthreads();

    for (int s = 1024/2; s>=1; s/=2){ //shared mem
        if (threadIdx.x < s){
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }
     if (threadIdx.x == 0) {
        atomicAdd(res, shmem[0]);
     }
}

template <typename T>
__global__ void sh_mem_atomicv2(T *a, T* b, T *res, int n){ //K3
    // instead of just copy to shared mem, do one set of reduction as well with copy
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (idx >= n) return;
    __shared__ T shmem[1024]; //using 1024 thread blocks
    shmem[threadIdx.x] = a[idx]*b[idx] + a[idx+blockDim.x]*b[idx+blockDim.x];
    __syncthreads();

    for (int s = 1024/2; s>=1; s/=2){ //all shared mem reduce their 1024 threads
        if (threadIdx.x < s){
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }
     if (threadIdx.x == 0) {
        atomicAdd(res, shmem[0]);// atomic to reduce final shared memory values in each block to global value
     }
}

template <typename T, int warpsize = 32, int num_warps = 32> //threads in warp, number of warps in block
__global__ void warp_shuffle(T *a, T* b, T *res, int n){ //K4
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;//same as before, do operation while loading to registers
    if (idx >= n) return;
    //1024 thread launched with 32 threads in a warp
    //so 32 total warps, len(shared memory) must be 32 here so all 32 different warps can be copied into warp 0 for final reduction
    assert(num_warps == blockDim.x/warpsize);
    __shared__ T shmem[num_warps];
    T my_val; //for each thread
    my_val = a[idx]*b[idx] + a[idx+blockDim.x]*b[idx+blockDim.x];
    int warpID = threadIdx.x / warpsize; // which warp group am I 0 - 31
    int lane = threadIdx.x % warpsize; // in warp group X, which thread am I 0 - 31
    unsigned mask = 0xffffffff; //all thread in the warp will participate

    //all warps in block
    for (int offset=warpsize/2; offset>=1; offset>>=1){
        my_val += __shfl_down_sync(mask, my_val, offset);
    }
    if (lane==0){
        shmem[warpID] = my_val; //copy to shared mem to later transfer to warp 0 for final sum
    }
    __syncthreads();

    if ((warpID == 0)&&(threadIdx.x < 32)){ //32 threads (0-31) from warp 0 will participate here
        my_val = shmem[lane];
        
        for (int offset=warpsize/2; offset>=1; offset>>=1){
            my_val += __shfl_down_sync(mask, my_val, offset);
        }
    }

    if ((warpID==0)&&(threadIdx.x==0)){
        atomicAdd(res, my_val);
    }
    
}

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

int main() {

    using vec_t = double;
    vec_t *h_a, *d_a,*h_b, *d_b, *h_ab, *d_out, *h_out;
    h_a = new vec_t[N];
    h_b = new vec_t[N];
    h_ab = new vec_t[N];
    h_out = new vec_t[1];

    for (int i=0; i<N; i++){
        h_a[i] = std::sin(i)/100000;//std::remainder(static_cast<vec_t>(i),1000.0)/1000000.0;
        h_b[i] = std::cos(i)/100000;
        h_ab[i] = h_a[i]*h_b[i];
    }
    vec_t h_result = std::reduce(h_ab,h_ab+N);

    cudaMalloc((void**)&d_a, N*sizeof(vec_t));
    cudaMalloc((void**)&d_b, N*sizeof(vec_t));
    cudaMalloc((void**)&d_out, sizeof(vec_t));
    cudaMemset(d_out, 0, sizeof(vec_t));
    cudaMemcpy(d_a, h_a, N*sizeof(vec_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(vec_t), cudaMemcpyHostToDevice);
    cudaCheckErrors("Malloc and memcpy to device");

    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    dim3 gridSize(blocks,1,1);
    dim3 blockSize(threads,1,1);

    #ifdef K1
    atomic_dp<double><<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
    #endif

    #ifdef K2
    sh_mem_atomic<double><<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
    #endif

    #ifdef K3
    sh_mem_atomicv2<double><<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
    #endif

    #ifdef K4
    warp_shuffle<double, 32, threads/32><<<gridSize, blockSize>>>(d_a, d_b, d_out, N);
    #endif

    cudaCheckErrors("Kernel launch");
    cudaDeviceSynchronize();


    cudaMemcpy(h_out, d_out, sizeof(vec_t), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy back to host");

        if ((std::fabs(*h_out - h_result)/std::fabs(h_result)) > 1e-1f){
            std::cout << "Answer is wrong; ans =" << *h_out << "  expected: " << h_result << std::endl;
            return -1;
        } else {
            std::cout << "GPU result = " << *h_out << " CPU result = " << h_result << std::endl;
        }

    std::cout << "Finished!" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    cudaCheckErrors("cudaFree");
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;

    return 0;
}