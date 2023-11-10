# Assignment III: CUDA Basics II

<https://github.com/ysiad2001/DD2360HT23/tree/master/hw_3>

## Exercise 1 - Histogram and Atomics

The parallel histograming is achieved by having a copy of an input array in all kernels ``histogram_kernel``, each kernel takes the indexes corresponding to its thread ID. Whenever an element is found on the array, ``atomicAdd()`` is called to count the index. After all the counting procedure has stopped, another kernel ``convert_kernel`` is called to make all entries greater than 127 be 127. The major optimization is to parallelize the histogramming process into 1024 parallel threads.

Each thread will handle ``num_elements/NUM_THREADS_PER_BLOCK`` array indexes, each index reference will result in a call to ``*input`` and a call to ``*bin``. So there are ``2*num_elements/NUM_THREADS_PER_BLOCK`` global memory reads.

Each thread will handle ``num_elements/NUM_THREADS_PER_BLOCK`` atomic operations for histogramming, and ``NUM_BINS/NUM_THREADS_PER_BLOCK`` for adding shared memory to global memory.

The only shared memory is ``__shared__ unsigned int shared_bins[NUM_BINS]``

If a lot of array indexes are at the same value, then it is likely that more threads will try to access the same index of the ``shared_bin``, thus creating contention.

The input length 512, 8 bins

![image](./ex1.png)

In the report, the shared memory configuration size is 32.77Kb and the achieved occupancy is 29.74\%. Nvsight produced the performance warning as shown here
```
This kernel's theoretical occupancy is not impacted by any block limit. The difference between calculated theoretical (100.0%) and measured achieved occupancy (29.7%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can occur between warps within a block as well as across blocks of the same kernel.
```
## Exercise 2 - A Particle Simulation Application

The code was compiled and executed on my local PC, made no change to the Makefile.

```bash
cd ./sputniPIC
mkdir data
make
./bin/sputniPIC.out ./inputfiles/GEM_2D.inp 
```

In mover_PC_gpu(), the functionality can be parallelized by having ``part->nop`` threads and simulating one particle on one kernel each time. The code inside the for loop ``for (int i = 0; i < part->nop; i++)`` can be put into the kernel function, along with all the variables needed.

Inside the kernel function, all the triple pointer variables need to be replaced with single pointers. For example, ``grd->XN[ix - 1][iy][iz]`` should be replaced with ``grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)]``.

Outside the kernel function, the memory to be copied from host to device is a struct with multiple arrays, so it is needed to have customized memory allocation and deallocation functions for each of the structs.

The execution time of CPU implementation is
```
   Tot. Simulation Time (s) = 46.2174
   Mover Time / Cycle   (s) = 1.73421
   Interp. Time / Cycle (s) = 2.74868
```

The execution time of GPU implementation is
```
    Tot. Simulation Time (s) = 30.9217
   Mover Time / Cycle   (s) = 0.192811
   Interp. Time / Cycle (s) = 2.75575
```