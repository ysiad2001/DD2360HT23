#include <cuda_runtime.h>
#include <iostream>
using namespace std;
#define CUDA_CALL(F)                                    \
    if ((F) != cudaSuccess)                             \
    {                                                   \
        printf("Error at line %d: %s\n", __LINE__,      \
               cudaGetErrorString(cudaGetLastError())); \
        exit(-1);                                       \
    };

// Define the struct
struct my_list
{
    int num;  // number of element in the list
    int *arr; // the list is dynamically allocated
};

__global__ void test_kernel(struct my_list *d_list)
{
    // Random kernel function
    atomicAdd(&d_list->arr[1], 114514);
    atomicAdd(&d_list->arr[2], 1919810);
}

void cudaMemcpyListHostToDevice(my_list **d_list, my_list *h_list)
{
    // Initiate the h_listCopy for later use
    my_list dummyList;
    my_list *h_listCopy = &dummyList;

    // Shallow Copy the content of h_list to h_listCopy
    *h_listCopy = *h_list;

    // Allocate the device variables
    int *d_arr;
    CUDA_CALL(cudaMalloc((void **)&d_arr, sizeof(int) * h_list->num));
    CUDA_CALL(cudaMalloc((void **)d_list, sizeof(my_list)));

    // Copy the array from host to device
    CUDA_CALL(cudaMemcpy(d_arr, h_list->arr, sizeof(int) * h_list->num, cudaMemcpyHostToDevice));

    // Redirect the h_listCopy->arr pointer to a device array
    h_listCopy->arr = d_arr;

    // Copy the h_listCopy from host to device
    CUDA_CALL(cudaMemcpy(*d_list, h_listCopy, sizeof(my_list), cudaMemcpyHostToDevice));
}

void cudaMemcpyListDeviceToHost(my_list *h_list, my_list **d_list)
{
    // Save a copy of h_listCopy, all pointers in there are allocated to host memory
    my_list dummyList;
    my_list *h_listCopy = &dummyList;

    *h_listCopy = *h_list;

    // Copy the d_list from device to host
    CUDA_CALL(cudaMemcpy(h_list, *d_list, sizeof(my_list), cudaMemcpyDeviceToHost));

    // Copy the array from device to host
    CUDA_CALL(cudaMemcpy(h_listCopy->arr, h_list->arr, sizeof(int) * h_list->num, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(h_list->arr));

    // Redirect the h_list->arr pointer to a host array
    h_list->arr = h_listCopy->arr;

    CUDA_CALL(cudaFree(*d_list));
}

int main()
{
    // Initialize the h_list needed to be processed on
    my_list *h_list = new my_list;
    h_list->num = 3;
    h_list->arr = new int[h_list->num];
    h_list->arr[0] = 0;
    h_list->arr[1] = 1;
    h_list->arr[2] = 2;

    my_list *d_list;
    cudaMemcpyListHostToDevice(&d_list, h_list);
    test_kernel<<<1, 1>>>(d_list);

    cudaMemcpyListDeviceToHost(h_list, &d_list);
    cout << h_list->num << endl;
    cout << h_list->arr[0] << endl;
    cout << h_list->arr[1] << endl;
    cout << h_list->arr[2] << endl;

    delete h_list;

    return 0;
}