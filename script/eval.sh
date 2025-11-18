#!/bin/bash

clear

cd ..

# Compilation avec OpenMP
cmake -B build -DUSE_OPENMP=ON
cmake --build build

# Compilation sans OpenMP
# cmake -B build -DUSE_OPENMP=OFF
# cmake --build build

echo "Evaluation des algorithmes"

cd ./build/src/
export OMP_NUM_THREADS=8 # Sets the number of threads to 4
./EVAL 8
# gdb ./EVAL 
cd ../../script