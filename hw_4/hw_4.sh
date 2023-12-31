echo compiling ex2 and ex3

nvcc ../hw_2/ex1.cu -o ex2_nonstream
nvcc ex2.cu -o ex2_stream
nvcc ex3.cu -o ex3 -lcusparse -lcublas

echo running ex2_nonstream with input from 1024 to 10240000
./ex2_nonstream 1024
./ex2_nonstream 10240
./ex2_nonstream 102400
./ex2_nonstream 1024000
./ex2_nonstream 10240000
./ex2_nonstream 102400000

echo running ex2_stream with input from 1024 to 10240000 and 4 streams
./ex2_stream 1024 4
./ex2_stream 10240 4
./ex2_stream 102400 4
./ex2_stream 1024000 4
./ex2_stream 10240000 4
./ex2_stream 102400000 4

echo running ex2_stream with input 102400000 and 4 streams and profiling
nvprof --output-profile ex2_stream_profile.nvprof ./ex2_stream 102400000 4

echo running ex2 with input 10240000 and 4 to 8 streams
./ex2_stream 10240000 4
./ex2_stream 10240000 5
./ex2_stream 10240000 6
./ex2_stream 10240000 7
./ex2_stream 10240000 8

echo running ex3 with dimX=128 nsteps = 100, 500, 1000, 5000, 10000 with prefetching
./ex3 128 100 1
./ex3 128 500 1
./ex3 128 1000 1
./ex3 128 5000 1
./ex3 128 10000 1

echo running ex3 with dimX=128 nsteps = 100, 500, 1000, 5000, 10000 without prefetching
./ex3 128 100 0
./ex3 128 500 0
./ex3 128 1000 0
./ex3 128 5000 0
./ex3 128 10000 0