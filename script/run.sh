#!/bin/bash

clear

## compile 
cd ..

# Compilation avec OpenMP
cmake -B build -DUSE_OPENMP=ON
cmake --build build

# make --build build --clean-first


echo "Execution du CNN ..."

cd ./build/src/
export OMP_NUM_THREADS=4 # Sets the number of threads to 4
./CNN 4
# gdb ./CNN
cd ../../script


# execute CNN
# ./src/CNN

# cd ../script



# ./src/CNN --dataset path/to/data --epochs 50