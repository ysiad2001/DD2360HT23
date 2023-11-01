echo compiling ex1

nvcc ex1.cu -o ex1

/usr/local/cuda-11/bin/nv-nsight-cu-cli ./ex1 1024