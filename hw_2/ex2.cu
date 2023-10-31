
#include <stdio.h>
#include <sys/time.h>
#include <cassert>
#include <iostream>

#define DataType float
#define ERROR_MARGIN (0.0001)

#define NUM_THREAD_X (32)
#define NUM_THREAD_Y (32)

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                     int numAColumns, int numBRows, int numBColumns)
{
    // Matrix multiplication
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if ((column >= numBColumns) || (row >= numARows))
        return;

    for (unsigned i = 0; i < numAColumns; i++)
    {
        C[row * numBColumns + column] += A[row * numAColumns + i] * B[i * numBColumns + column];
    }
}

// Random float from fMin to fMax
DataType fRand(DataType fMin, DataType fMax)
{
    DataType f = (DataType)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main(int argc, char **argv)
{

    struct timeval time1;
    struct timeval time2;
    struct timeval time3;
    struct timeval time4;

    DataType *hostA;     // The A matrix
    DataType *hostB;     // The B matrix
    DataType *hostC;     // The output C matrix
    DataType *resultRef; // The reference result
    DataType *deviceA;
    DataType *deviceB;
    DataType *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;
    int numCColumns;

    //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

    assert(argc == 4);
    numARows = atoi(argv[1]);
    numAColumns = atoi(argv[2]);
    numBRows = numAColumns;
    numBColumns = atoi(argv[3]);
    numCRows = numARows;
    numCColumns = numBColumns;

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    // Allocate Host memory for input and output
    hostA = (DataType *)malloc(sizeof(DataType) * numARows * numAColumns);
    hostB = (DataType *)malloc(sizeof(DataType) * numBRows * numBColumns);
    hostC = (DataType *)malloc(sizeof(DataType) * numCRows * numCColumns);
    resultRef = (DataType *)malloc(sizeof(DataType) * numCRows * numCColumns);
    // Initialize hostA and hostB to random numbers, and create reference result in CPU
    for (int i = 0; i < numARows; i++)
    {
        for (int j = 0; j < numAColumns; j++)
        {
            hostA[i * numAColumns + j] = fRand(0, 1);
        }
    }

    for (int i = 0; i < numBRows; i++)
    {
        for (int j = 0; j < numBColumns; j++)
        {
            hostB[i * numBColumns + j] = fRand(0, 1);
        }
    }

    for (int i = 0; i < numCRows; i++)
    {
        for (int j = 0; j < numCColumns; j++)
        {
            resultRef[i * numCColumns + j] = 0;
            for (int k = 0; k < numAColumns; k++)
            {
                resultRef[i * numCColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            }
        }
    }

    // Allocate GPU memory
    cudaMalloc(&deviceA, sizeof(DataType) * numAColumns * numARows);
    cudaMalloc(&deviceB, sizeof(DataType) * numBColumns * numBRows);
    cudaMalloc(&deviceC, sizeof(DataType) * numCColumns * numCRows);

    gettimeofday(&time1, NULL);

    // Copy memory to the GPU 
    cudaMemcpy(deviceA, hostA, sizeof(DataType) * numAColumns * numARows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(DataType) * numBColumns * numBRows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC, hostC, sizeof(DataType) * numCColumns * numCRows, cudaMemcpyHostToDevice);

    // Initialize the grid and block dimensions
    int numThreadX = NUM_THREAD_X;
    int numThreadY = NUM_THREAD_Y;
    int numBlockX = (numCColumns + NUM_THREAD_X - 1) / NUM_THREAD_X;
    int numBlockY = (numCRows + NUM_THREAD_Y - 1) / NUM_THREAD_Y;

    gettimeofday(&time2, NULL);
    // Launch the GPU Kernel
    gemm<<<dim3(numBlockX, numBlockY, 1), dim3(numThreadX, numThreadY, 1)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    gettimeofday(&time3, NULL);

    // Copy the GPU memory back to the CPU
    cudaMemcpy(hostC, deviceC, sizeof(DataType) * numCColumns * numCRows, cudaMemcpyDeviceToHost);
    gettimeofday(&time4, NULL);

    double duration;
    duration = time2.tv_sec * 1000000 + time2.tv_usec - (time1.tv_sec * 1000000 + time1.tv_usec);
    std::cout << "Host to Device copy time is " << duration << std::endl;

    duration = time3.tv_sec * 1000000 + time3.tv_usec - (time2.tv_sec * 1000000 + time2.tv_usec);
    std::cout << "Kernel time is " << duration << std::endl;

    duration = time4.tv_sec * 1000000 + time4.tv_usec - (time3.tv_sec * 1000000 + time3.tv_usec);
    std::cout << "Host to Device copy time is " << duration << std::endl;

    // Compare the output with the reference

    for (int i = 0; i < numCRows; i++)
    {
        for (int j = 0; j < numCColumns; j++)
        {
            if (
                (hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) > (ERROR_MARGIN) ||
                (hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) < -(ERROR_MARGIN))
                printf("Error in output[%d][%d] by %f\n", i, j, (hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]));
        }
    }

    // Cleanup
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
