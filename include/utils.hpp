#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "convolution.hpp"

using namespace std;
using namespace Eigen;
MatrixXd one_hot(const VectorXd& y, int num_labels=0);
void logCNNArchitecture(const ImageDataset& imgDataset, 
                              const ConvLayer& conv1, const PoolLayer& pool1,
                              const ConvLayer& conv2, const PoolLayer& pool2,
                              int image_size, int input_channels, int n_images,
                              const vector<int>& dense_architecture = {64, 32});

void displayPredictions(const MatrixXd& predictions, 
                       const vector<int>& true_labels, 
                       const VectorXd& Y_encoded,
                       int num_samples = 5) ;


#endif 