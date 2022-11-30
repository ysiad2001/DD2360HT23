start=`date +%s.%N`
echo 
echo $PWD$
cd ./openmp/bfs
./run
end=`date +%s.%N`
runtime_openmp_bfs=$( echo "$end - $start" | bc -l )

start=`date +%s.%N`
echo 
cd ..
cd ..
cd ./cuda/bfs
echo $PWD$
./run
end=`date +%s.%N`
runtime_cuda_bfs=$( echo "$end - $start" | bc -l )


start=`date +%s.%N`
echo 
cd ..
cd ..
cd ./openmp/hotspot
echo $PWD$
./run
end=`date +%s.%N`
runtime_openmp_hotspot=$( echo "$end - $start" | bc -l )


start=`date +%s.%N`
echo 
cd ..
cd ..
cd ./cuda/hotspot
echo $PWD$
./run
end=`date +%s.%N`
runtime_cuda_hotspot=$( echo "$end - $start" | bc -l )


start=`date +%s.%N`
echo 
cd ..
cd ..
cd ./openmp/lud
echo $PWD$
./run
end=`date +%s.%N`
runtime_openmp_lud=$( echo "$end - $start" | bc -l )


start=`date +%s.%N`
echo 
cd ..
cd ..
cd ./cuda/lud
echo $PWD$
./run
end=`date +%s.%N`
runtime_cuda_lud=$( echo "$end - $start" | bc -l )

echo openmp bfs $runtime_openmp_bfs
echo openmp hotspot $runtime_openmp_hotspot
echo openmp lud $runtime_openmp_lud
echo cuda bfs $runtime_cuda_bfs
echo cuda hotspot $runtime_cuda_hotspot
echo cuda lud $runtime_cuda_lud
