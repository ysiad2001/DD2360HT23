#include "../cuda_utils.h"

#define NUM_THREAD 256 // cannot be larger than 1024

__global__ void vecAdd(int *a, int *b, int *c, int inputSize)
{
    unsigned thisIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(thisIndex < inputSize)
        c[thisIndex] = a[thisIndex] + b[thisIndex];
}

// Used to handle cuda error
#define CUDA_CALL(F)                                    \
    if ((F) != cudaSuccess)                             \
    {                                                   \
        printf("Error at line %d: %s\n", __LINE__,      \
               cudaGetErrorString(cudaGetLastError())); \
        exit(-1);                                       \
    };

int main(int argc, char **argv)
{
    int inputSize = atoi(argv[1]);
    // timing related
    struct timeval startTime;
    struct timeval endTime;

    // host copies of variables a, b & c
    int *a, *b, *c, *c_ref;

    // device copies of variables a, b & c
    int *d_a, *d_b, *d_c;
    int size = sizeof(int);

    // Allocate space for host copies of a, b, c
    a = (int *)malloc(size * inputSize);
    b = (int *)malloc(size * inputSize);
    c = (int *)malloc(size * inputSize);
    c_ref = (int *)malloc(size * inputSize);

    // Allocate space for device copies of a, b, c
    CUDA_CALL(cudaMalloc((void **)&d_a, size * inputSize));
    CUDA_CALL(cudaMalloc((void **)&d_b, size * inputSize));
    CUDA_CALL(cudaMalloc((void **)&d_c, size * inputSize));

    // Setup input values and benchmark
    for (unsigned i = 0; i < inputSize; ++i)
    {
        a[i] = rand();
        b[i] = rand();
        c_ref[i] = a[i] + b[i];
    }
    // Copy inputs to device
    START_TIMER;
    CUDA_CALL(cudaMemcpy(d_a, a, size * inputSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_b, b, size * inputSize, cudaMemcpyHostToDevice));
    END_TIMER;
    PRINT_TIMER("Host to Device time");

    // Launch vecAdd() kernel on GPU
    START_TIMER;
    vecAdd<<<(inputSize/NUM_THREAD) + 1, NUM_THREAD>>>(d_a, d_b, d_c,inputSize);
    cudaDeviceSynchronize();
    END_TIMER;
    PRINT_TIMER("Kernel time");

    // Copy result back to host
    START_TIMER;
    CUDA_CALL(cudaMemcpy(c, d_c, size * inputSize, cudaMemcpyDeviceToHost));
    END_TIMER;
    PRINT_TIMER("Device to Host time");

    // Reference check
    for (unsigned i = 0; i < inputSize; ++i)
    {
        if (c_ref[i] != c[i])
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
    free(c_ref);

    return 0;
}