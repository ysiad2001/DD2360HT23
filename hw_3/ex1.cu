
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <cassert>

#define DataType double
#define ERROR_MARGIN (0.00001)
#define NUM_BLOCK (1)
#define NUM_THREAD (256)

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
    //@@ Insert code to implement vector addition here
    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        out[i] = in1[i] + in2[i];
    }
}

//@@ Insert code to implement timer start

//@@ Insert code to implement timer stop

int main(int argc, char **argv)
{
    struct timeval time1;
    struct timeval time2;
    struct timeval time3;
    struct timeval time4;

    

    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    //@@ Insert code below to read in inputLength from args
    assert(argc == 2);
    inputLength = atoi(argv[1]);
    printf("The input length is %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output
    hostInput1 = (double *)malloc(sizeof(DataType) * inputLength);
    hostInput2 = (double *)malloc(sizeof(DataType) * inputLength);
    hostOutput = (double *)malloc(sizeof(DataType) * inputLength);
    resultRef = (double *)malloc(sizeof(DataType) * inputLength);
    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (unsigned i = 0; i < inputLength; ++i)
    {
        hostInput1[i] = fRand(0, 1);
        hostInput2[i] = fRand(0, 1);
        hostOutput[i] = 0;
        resultRef[i] = hostInput1[i] + hostInput2[i];
    }
    
    //@@ Insert code below to allocate GPU memory here
    cudaMalloc(&deviceInput1, sizeof(DataType) * inputLength);
    cudaMalloc(&deviceInput2, sizeof(DataType) * inputLength);
    cudaMalloc(&deviceOutput, sizeof(DataType) * inputLength);

    gettimeofday(&time1, NULL);

    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceOutput, hostOutput, sizeof(DataType) * inputLength, cudaMemcpyHostToDevice);

    //@@ Initialize the 1D grid and block dimensions here
    int numBlock = NUM_BLOCK;
    int numThread = NUM_THREAD;

    gettimeofday(&time2, NULL);

    //@@ Launch the GPU Kernel here
    vecAdd<<<numBlock, numThread>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    //@@ Copy the GPU memory back to the CPU here
    cudaDeviceSynchronize();
    gettimeofday(&time3, NULL);

    cudaMemcpy(hostOutput, deviceOutput, sizeof(DataType) * inputLength, cudaMemcpyDeviceToHost);
    gettimeofday(&time4, NULL);

    double duration = 0;
    duration = time2.tv_sec * 1000000 + time2.tv_usec - (time1.tv_sec * 1000000 + time1.tv_usec);
    std::cout << "Duration 1 is " << duration << std::endl;

    duration = time3.tv_sec * 1000000 + time3.tv_usec - (time2.tv_sec * 1000000 + time2.tv_usec);
    std::cout << "Duration 2 is " << duration << std::endl;

    duration = time4.tv_sec * 1000000 + time4.tv_usec - (time3.tv_sec * 1000000 + time3.tv_usec);
    std::cout << "Duration 3 is " << duration << std::endl;

    //@@ Insert code below to compare the output with the reference
    for (unsigned i = 0; i < inputLength; ++i)
    {
        if (resultRef[i] - hostOutput[i] > (ERROR_MARGIN) || resultRef[i] - hostOutput[i] < -(ERROR_MARGIN))
            printf("Error in output[%d] by %f\n", i, (resultRef[i] - hostOutput[i]));
    }

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
