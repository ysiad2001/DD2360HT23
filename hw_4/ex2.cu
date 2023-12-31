#include "../cuda_utils.h"

#define THREAD_PER_BLOCK 256 // cannot be larger than 1024

size_t numStreams = 4;
size_t segSize = 0;

__global__ void add(int *a, int *b, int *c,int segSize)
{
    unsigned threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < segSize)
    {
        c[threadId] = a[threadId] + b[threadId];
    }
}

int main(int argc, char **argv)
{
    int inputSize = atoi(argv[1]);
    numStreams = atoi(argv[2]);
    segSize = inputSize / numStreams;
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

    // Setup input values and reference
    for (unsigned i = 0; i < inputSize; ++i)
    {
        a[i] = rand();
        b[i] = rand();
        c_ref[i] = a[i] + b[i];
    }

    // Create streams
    START_TIMER;
    cudaStream_t streams[numStreams];
    size_t segSize = (inputSize + numStreams - 1) / numStreams;
    size_t segSizeByte = sizeof(int) * segSize;
    size_t lastStreamSize = inputSize - segSize * (numStreams - 1);
    size_t lastStreamSizeByte = sizeof(int) * lastStreamSize;
    for (int i = 0; i < numStreams; i++)
    {
        CUDA_CALL(cudaStreamCreate(&streams[i]));
    }
    END_TIMER;
    PRINT_TIMER("Create streams time");

    START_TIMER;
    int offset = 0;
    for (int i = 0; i < numStreams; i++)
    {
        offset = i * segSize;
        if (i == numStreams - 1)//last stream
        {
            CUDA_CALL(cudaMemcpyAsync(d_a + offset, a + offset, lastStreamSizeByte,cudaMemcpyHostToDevice, streams[i]));
            CUDA_CALL(cudaMemcpyAsync(d_b + offset, b + offset, lastStreamSizeByte,cudaMemcpyHostToDevice, streams[i]));
            CUDA_CALL(cudaStreamSynchronize(streams[i]));
            break;
        }
        CUDA_CALL(cudaMemcpyAsync(d_a + offset, a + offset, segSizeByte,cudaMemcpyHostToDevice, streams[i]));
        CUDA_CALL(cudaMemcpyAsync(d_b + offset, b + offset, segSizeByte,cudaMemcpyHostToDevice, streams[i]));
        CUDA_CALL(cudaStreamSynchronize(streams[i]));
    }
    END_TIMER;
    PRINT_TIMER("Host to Device Memcpy Time");

    START_TIMER;
    for (int i = 0; i < numStreams; i++)
    {
        offset = i * segSize;
        add<<<segSize / THREAD_PER_BLOCK + 1, THREAD_PER_BLOCK, 0, streams[i]>>>(d_a + offset, d_b + offset, d_c + offset,segSize);
    }
    CUDA_CALL(cudaDeviceSynchronize());
    END_TIMER;
    PRINT_TIMER("Kernel Time");


    START_TIMER;
    for (int i = 0; i < numStreams; i++)
    {
        offset = i * segSize;
        if (i == numStreams - 1)//last stream
        {
            CUDA_CALL(cudaMemcpyAsync(c + offset, d_c + offset, lastStreamSizeByte,cudaMemcpyDeviceToHost, streams[i]));
            CUDA_CALL(cudaStreamSynchronize(streams[i]));
            break;
        }
        CUDA_CALL(cudaMemcpyAsync(c + offset, d_c + offset, segSizeByte,cudaMemcpyDeviceToHost, streams[i]));
        CUDA_CALL(cudaStreamSynchronize(streams[i]));
    }
    END_TIMER;
    PRINT_TIMER("Device to Host Memcpy Time");

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