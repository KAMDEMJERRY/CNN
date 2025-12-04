#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <Eigen/Dense>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include "convolution.hpp"
#include "imgdataset.hpp"
#include "omp.hpp"


using namespace std;
using namespace Eigen;
MatrixXd one_hot(const VectorXd& y, int num_labels=0);
void logCNNArchitecture(const ImageDataset& imgDataset, 
                              const ConvLayer& conv1, const PoolLayer& pool1,
                              const ConvLayer& conv2, const PoolLayer& pool2,
                              int image_size, int input_channels, int n_images,
                              const vector<int>& dense_architecture = {64, 32});

std::unordered_map<std::string, std::string> loadEnvFile(const std::string &filename);

#endif 