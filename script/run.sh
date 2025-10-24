#!/bin/bash

clear

## compile 
cd ..
cmake --build build
# make --build build --clean-first


echo "Execution du CNN ..."

cd ./build/src/
gdb ./CNN
cd ../../script


# execute CNN
# ./src/CNN

# cd ../script



# ./src/CNN --dataset path/to/data --epochs 50