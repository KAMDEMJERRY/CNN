#!/bin/bash

clear

cd ..


# export OMP_NUM_THREADS = $(nproc)
# export OMP_NUM_THREADS = 2


# Compilation avec OpenMP
cmake -B build -DUSE_OPENMP=ON
cmake --build build

# Compilation sans OpenMP
# cmake -B build -DUSE_OPENMP=OFF