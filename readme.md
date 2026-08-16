# FILES:

| **FILENAME**                             | **DESCRIPTION**                                              |
| ---------------------------------------- | ------------------------------------------------------------ |
| `l1_Hello_from_thread.cu`                | Hello World equivalent for CUDA                              |
| `l2_vector_addition.cu`                  | Vector addition with CUDA error checking <br /> 1D threads and blocks |
| `l3_naive_matmul_and_profiling.cu`       | Matrix multiplication with profiling using CUDA events and also `nvtx` so saved profile can be opened in Nsight compute.`nvtx` inside preprocessor directives to avoid computational overhead when not profiling. |
| `l4_matmul_using_cuBLAS.cu`              | matrix multiplication using cuBLAS. NOTES: cuBLAS is similar to BLAS & Fortran and is column major. The order of matrix need to be switched because $(B^TA^T)^T=(A^T)^T(B^T)^T=AB$. |
| `l5_array_sum_atomics.cu`                | atomics to get sum of array. Despite being memory safe, atomics is serial and should be combined with other techniques such as reduction (next part). |
| `l5_array_sum_reduction_warp_shuffle.cu` | on hold                                                      |
| `l6_openACC_SAXPY.cpp`                   | Detour: using openACC, we can easily parallelize existing code that runs either on gpu/cpu depending on available hardware or parts of the same code can run on gpus and cpus. We need to install `NVIDIA_HPC_SDK` first and update the bash path.<br /> Use nvc++ -acc -mp flags to use openACC and openMP <br /> The GPU time for this code is worse than CPU time most likely due to large memory transfer time and low computation. We can see this on `l6_profile.txt`, line 157 as well. |
| `l7_transient_heat_equation`             | Transient heat equation for conduction solved on CPU for baseline and then using openMP, CUDA (via openACC) and only CUDA. |
| `multiGPU_ThermalConduction`             | Thermal conduction problem for multiGPU setup using CUDA aware MPI. Streams are used to try to hide communication latency during computation of non-halo grid points. |
| `l8_shared_mem_matmul.cu`                | Matrix multiplication using shared memory                    |
| `l9_matmul_with_1D_Blocktiling.cu`       | Matmul using 1D blocktiling and shared memory                |
| `dot_product.cu`                         | Dot product of two vector using atomic, shared memory and warp shuffle. Kernel run time on 5090 laptop GPU = 2.33 ms, 83.07 us, 52.13 us, 40,13 us corresponding to K1, K2, K3 and K4 in the code. |



# Resources:

[Nvidia CUDA Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

[Nvidia cuBLAS Doc](https://docs.nvidia.com/cuda/cublas/)

Tutorials by NVIDIA & Oak Ridge National Laboratory
