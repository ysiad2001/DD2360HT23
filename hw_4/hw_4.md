# Assignment IV: Advanced CUDA

<https://github.com/ysiad2001/DD2360HT23/tree/master/hw_4>

## Exercise 1 - Thread Scheduling and Execution Efficiency 

### X=800 Y=600 16X16 grid

The program should generate ``ceil(800/16)*ceil(600/16)=1900`` blocks. Each block has ``16*16`` threads with size ``32`` warps. In total ``1900*16*16/32=15200`` warps.

The blocks with block index ``(x,y)`` should handle threads with ``threadIdx.x`` range from  ``16*x`` to ``16*x+15`` ``threadIdx.y`` range from ``16*y`` to ``16*y+15``. The only blocks that have threads out of range are those with ``blockIdx.y==ceil(600/16)-1``. All of these blocks have 128 threads out of range and another 128 in range. Since they are in y-major alignment, each of these blocks will have ``128/32=4`` warps following the ``if`` statement and the other ``128/32=4`` follow the ``else`` statement. None of the warps have control divergence.

### X=600 Y=800 16X16 grid

In this case, blocks with ``blockIdx.x==ceil(600/16)-1`` will have control divergence in all warps. In total ``ceil(800/16)=50`` blocks with ``50*16*16/32=400`` warps.

### X=600 Y=799 16X16 grid

The situation is similar to the last question, blocks with ``blockIdx.x==ceil(600/16)-1`` will have control divergence in all warps. In total ``ceil(800/16)=50`` blocks with ``50*16*16/32=400`` warps.

## Exercise 2 - CUDA Streams

## Exercise 3 - Heat Equation with using NVIDIA libraries