echo compiling ex2 and ex3

nvcc ../hw_2/ex1.cu -o ex2_nonstream
nvcc ex2.cu -o ex2_stream
# nvcc ex3.cu -o ex3 -lcusparse -lcublas
echo running ex2_nonstream with input from 1024 to 10240000
./ex2_nonstream 1024
./ex2_nonstream 10240
./ex2_nonstream 102400
./ex2_nonstream 1024000
./ex2_nonstream 10240000
./ex2_nonstream 102400000

echo running ex2_stream with input from 1024 to 10240000
./ex2_stream 1024
./ex2_stream 10240
./ex2_stream 102400
./ex2_stream 1024000
./ex2_stream 10240000
./ex2_stream 102400000