#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define N 131070
#define NUM_THREAD 256 // cannot be larger than 1024

/*
Kernel function,
kernel 0 does a[0]+b[0]=c[0]; a[NUM_THREAD]+b[NUM_THREAD]=c[NUM_THREAD]; a[2*NUM_THREAD]+b[2*NUM_THREAD]=c[2*NUM_THREAD];...
kernel 1 does a[1]+b[1]=c[1]; a[NUM_THREAD+1]+b[NUM_THREAD+1]=c[NUM_THREAD+1]; a[2*NUM_THREAD+1]+b[2*NUM_THREAD+1]=c[2*NUM_THREAD+1];...
So all entries of the array a[i]+b[i]=c[i] are computed
*/
__global__ void add(int *a, int *b, int *c)
{
  unsigned thisIndex = threadIdx.x;
  while (thisIndex < N)
  {
    c[thisIndex] = a[thisIndex] + b[thisIndex];
    thisIndex += NUM_THREAD;
  }
}

// Used to handle cuda error
#define CUDA_CALL(F)                                \
  if ((F) != cudaSuccess)                           \
  {                                                 \
    printf("Error at line %d: %s\n", __LINE__,      \
           cudaGetErrorString(cudaGetLastError())); \
    exit(-1);                                       \
  };

int main()
{
  // timing related
  struct timeval startTime;
  struct timeval endTime;

  // host copies of variables a, b & c
  int *a, *b, *c, *c_benchmark;

  // device copies of variables a, b & c
  int *d_a, *d_b, *d_c;
  int size = sizeof(int);

  // Allocate space for host copies of a, b, c
  a = (int *)malloc(size * N);
  b = (int *)malloc(size * N);
  c = (int *)malloc(size * N);
  c_benchmark = (int *)malloc(size * N);

  // Allocate space for device copies of a, b, c
  CUDA_CALL(cudaMalloc((void **)&d_a, size * N));
  CUDA_CALL(cudaMalloc((void **)&d_b, size * N));
  CUDA_CALL(cudaMalloc((void **)&d_c, size * N));

  // Setup input values and benchmark
  for (unsigned i = 0; i < N; ++i)
  {
    a[i] = rand();
    b[i] = rand();
    c_benchmark[i] = a[i] + b[i];
  }

  // Copy inputs to device
  gettimeofday(&startTime, NULL);
  CUDA_CALL(cudaMemcpy(d_a, a, size * N, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_b, b, size * N, cudaMemcpyHostToDevice));
  gettimeofday(&endTime, NULL);
  printf("Host to Device time %d\n", (int)(endTime.tv_sec * 1000000 + endTime.tv_usec -
                                           (startTime.tv_sec * 1000000 + startTime.tv_usec)));
  // Launch add() kernel on GPU
  gettimeofday(&startTime, NULL);
  add<<<1, NUM_THREAD>>>(d_a, d_b, d_c);
  gettimeofday(&endTime, NULL);
  printf("Kernel time %d\n", (int)(endTime.tv_sec * 1000000 + endTime.tv_usec -
                                   (startTime.tv_sec * 1000000 + startTime.tv_usec)));
  // Copy result back to host
  gettimeofday(&startTime, NULL);
  CUDA_CALL(cudaMemcpy(c, d_c, size * N, cudaMemcpyDeviceToHost));
  gettimeofday(&endTime, NULL);
  printf("Device to Host time %d\n", (int)(endTime.tv_sec * 1000000 + endTime.tv_usec -
                                           (startTime.tv_sec * 1000000 + startTime.tv_usec)));
  for (unsigned i = 0; i < N; ++i)
  {
    if (c_benchmark[i] != c[i])
    {
      printf("Calculation wrong at a[%d]+b[%d]=c[%d] %d + %d = %d, aborting.\n", i, i, i, a[i], b[i], c[i]);
      break;
    }
  }

  printf("Calculation finished.\n");
  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(a);
  free(b);
  free(c);
  free(c_benchmark);

  return 0;
}