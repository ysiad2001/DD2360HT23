
#include <stdio.h>
#include <sys/time.h>
#include <random>

#define NUM_BINS (8)

#define NUM_BLOCKS (1)// This can only be 1, having multiple blocks will cause unsynced shared memory between blocks
#define NUM_THREADS_PER_BLOCK (1024)
#define NUM_THREADS (NUM_THREADS_PER_BLOCK * NUM_BLOCKS)

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins, unsigned int num_elements, unsigned int num_bins)
{

    int threadId = blockIdx.x * blockDim.x + threadIdx.x; // thread ID

    __shared__ unsigned int shared_bins[NUM_BINS];

    __syncthreads();
    for (unsigned binId = threadId; binId < NUM_BINS; binId += NUM_THREADS)
    {
        shared_bins[binId] = 0;
    }
    __syncthreads();
    for (unsigned inputId = threadId; inputId < num_elements; inputId += NUM_THREADS)
    {
        atomicAdd(&shared_bins[input[inputId]], 1);
    }
    __syncthreads();
    for (unsigned binId = threadId; binId < NUM_BINS; binId += NUM_THREADS)
    {
        atomicAdd(&bins[binId], shared_bins[binId]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned binId = threadId; binId < NUM_BINS; binId += NUM_THREADS)
    {
        if (bins[binId] > 127)
            bins[binId] = 127;
    }
}

int main(int argc, char **argv)
{

    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    // Read in inputLength from args
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);

    // Allocate Host memory for input and output
    hostInput = (unsigned int *)malloc(sizeof(unsigned int) * inputLength);
    hostBins = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BINS);

    // Initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    for (int i = 0; i < inputLength; i++)
    {
        hostInput[i] = rand() % NUM_BINS;
    }

    // Create reference result in CPU
    resultRef = (unsigned int *)malloc(sizeof(unsigned int) * NUM_BINS);
    for (int i = 0; i < NUM_BINS; i++)
    {
        resultRef[i] = 0;
        hostBins[i] = 0;
    }
    for (int i = 0; i < inputLength; i++)
    {
        ++resultRef[hostInput[i]];
    }
    for (int i = 0; i < NUM_BINS; i++)
    {
        if (resultRef[i] > 127)
            resultRef[i] = 127;
    }

    // Allocate GPU memory
    cudaMalloc((void **)&deviceInput, sizeof(unsigned int) * inputLength);
    cudaMalloc((void **)&deviceBins, sizeof(unsigned int) * NUM_BINS);

    // Copy memory to the GPU
    cudaMemcpy(deviceInput, hostInput, sizeof(unsigned int) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBins, hostBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyHostToDevice);

    // Launch the GPU Kernel
    histogram_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

    // Launch the second GPU Kernel
    convert_kernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(deviceBins, NUM_BINS);

    // Copy the GPU memory back to the CPU
    cudaMemcpy(hostBins, deviceBins, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);

    // Compare the output with the reference
    bool correctFlag = true;
    for (int i = 0; i < NUM_BINS; i++)
    {
        if (hostBins[i] != resultRef[i])
        {
            correctFlag = false;
            printf("The results are INCORRECT, at position %d, hostBins[i] = %d, resultRef[i] = %d.\n",i,hostBins[i],resultRef[i]);
            break;
        }
        printf("%d,",hostBins[i]);
    }
    if (correctFlag)
        printf("The results are CORRECT.\n");

    // Cleanup
    cudaFree(deviceInput);
    cudaFree(deviceBins);
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}
