#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>

//#define TESTING

#ifdef TESTING
const size_t size = 10; // not counting global boundary points
constexpr int iter = 5000;
#else
const size_t size = 50000; // not counting global boundary points
constexpr int iter = 50000;
#endif
// const size_t size2 = size * size;
// const int width_par_boundary = 2;

#define idx(r, c, width)                                                       \
  ((r)*width + (c)) // NOTE the brackets and REMEMBER TO ALWAYS USE IT.

// from NVIDIA, for error checking
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      fprintf(stderr, "*** FAILED - ABORTING\n");                              \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int matrixToFile(float *T, int myWidth, int myHeight,
                 const std::string &filename) {
#ifdef TESTING
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return 1;
  }
  for (int j = 0; j < myHeight; ++j) {
    for (int i = 0; i < myWidth; ++i) {
      file << std::setw(8) << std::fixed << std::setprecision(4)
           << T[idx(j, i, myWidth)] << " ";
    }
    file << std::endl;
  }
  file.close();
#endif
  return 0;
}

__global__ void ConductionGaussSidel(float *T, size_t _sizex, size_t _sizey) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x; // idx is taken
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  if (tx > 0 && tx < _sizex - 1 && ty > 0 && ty < _sizey - 1) {
    // clang-format off
        T[idx(ty, tx, _sizex)] = 0.25f * (T[idx(ty - 1, tx    , _sizex)] + 
                                          T[idx(ty + 1, tx    , _sizex)] +
                                          T[idx(ty    , tx - 1, _sizex)] + 
                                          T[idx(ty    , tx + 1, _sizex)]);
    // clang-format on
  }
}

int main(int argc, char *argv[]) {
  int myRank, nRank;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRank);
  size_t mysize_x = size + 2;
  size_t mysize_y = (size / nRank) + 2; // for halo exchange
  size_t mysize2 = mysize_x * mysize_y;
  float *h_T = new float[mysize2](); // initialized to zero

  for (int i = 0; i < mysize_y; ++i) // left right for all ranks
  {
    h_T[idx(i, 0, mysize_x)] = 100.0f;
    h_T[idx(i, mysize_x - 1, mysize_x)] = 200.0f;
  }
  if (myRank == 0) // bottom for 0 only
  {
    for (int j = 0; j < mysize_x; ++j) {
      h_T[idx(0, j, mysize_x)] = 300.0f;
    }
  }
  if (myRank == nRank - 1) // top for last rank
  {
    for (int j = 0; j < mysize_x; ++j) {
      h_T[idx(mysize_y - 1, j, mysize_x)] = 400.0f;
    }
  }

  int bottom_neighbor = (myRank<nRank-1) ? myRank +1 : MPI_PROC_NULL ;//(myRank + 1); // rank 0 is on top
  int top_neighbor = (myRank>0) ? myRank-1 : MPI_PROC_NULL;//myRank - 1;

  float *d_T;
  cudaMalloc(&d_T, mysize2 * sizeof(float));
  cudaMemcpy(d_T, h_T, mysize2 * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("Allocated d_T and copied data to device");

  const int threadsPerBlockx = 32;
  const int threadsPerBlocky = 32;
  const int blocksPerGridx =
      (mysize_x + threadsPerBlockx - 1) / threadsPerBlockx;
  const int blocksPerGridy =
      (mysize_y + threadsPerBlocky - 1) / threadsPerBlocky;
  dim3 blockDim(threadsPerBlockx, threadsPerBlocky);
  dim3 gridDim(blocksPerGridx, blocksPerGridy);

  auto start = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iter; ++i) {
    ConductionGaussSidel<<<gridDim, blockDim>>>(d_T, mysize_x, mysize_y);
    cudaDeviceSynchronize();
    // clang-format off
    //if (myRank != nRank - 1) {//send 2nd last row to top and receive into last row, but last rank last row is boundary
      MPI_Sendrecv(&d_T[idx(mysize_y - 2, 0, mysize_x)], mysize_x, MPI_FLOAT, bottom_neighbor, 0, 
                   &d_T[idx(mysize_y - 1, 0, mysize_x)], mysize_x, MPI_FLOAT, bottom_neighbor, 0, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //}

    //if (myRank != 0) {// Send 2nd row to bottom, receive into 1st row, but rank 0 row 0 is boundary
      MPI_Sendrecv(&d_T[idx(1, 0, mysize_x)], mysize_x, MPI_FLOAT,top_neighbor, 0, 
                   &d_T[idx(0, 0, mysize_x)], mysize_x, MPI_FLOAT, top_neighbor, 0, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //}
    // clang-format on
  }
  cudaDeviceSynchronize();
  auto end = MPI_Wtime(); // std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  double avg_time;
  MPI_Reduce(&duration, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (myRank == 0) {
    std::cout << "Average Kernel Execution time: " << duration / nRank << " s"
              << std::endl;
  }
  cudaCheckErrors("Kernel launch");

  cudaMemcpy(h_T, d_T, mysize2 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Copied data back to host");

  std::string filename = "output_rank_" + std::to_string(myRank) + ".txt";
  matrixToFile(h_T, mysize_x, mysize_y, filename);

  delete[] h_T;
  cudaFree(d_T);

  MPI_Finalize();

  return 0;
}