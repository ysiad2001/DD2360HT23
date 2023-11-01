
echo compiling ex1 and ex2
nvcc ./ex1.cu -o ex1
nvcc ./ex2.cu -o ex2

echo running nsight on ex1 with input 1024
/usr/local/cuda-11/bin/nv-nsight-cu-cli ./ex1 1024

echo running nsight on ex1 with input 131070
/usr/local/cuda-11/bin/nv-nsight-cu-cli ./ex1 131070

echo running ex1 with input from 1024 to 10240000
./ex1 1024
./ex1 10240
./ex1 102400
./ex1 1024000
./ex1 10240000

echo running ex2 with input 128 128 128
./ex2 128 128 128

echo running nsight on ex2 with input 128 128 128
/usr/local/cuda-11/bin/nv-nsight-cu-cli ./ex2 128 128 128

echo running ex2 with input 511 1023 4094
./ex2 511 1023 4094

echo running nsight on ex2 with input 511 1023 4094
/usr/local/cuda-11/bin/nv-nsight-cu-cli ./ex2 511 1023 4094