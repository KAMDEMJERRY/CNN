#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <string>

#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include "convolution.hpp"
#include "imgdataset.hpp"
#include "omp.hpp"

using namespace std;
using namespace Eigen;
MatrixXd one_hot(const VectorXd &y, int num_labels = 0);
void logCNNArchitecture(const ImageDataset &imgDataset,
                        const ConvLayer &conv1, const PoolLayer &pool1,
                        const ConvLayer &conv2, const PoolLayer &pool2,
                        int image_size, int input_channels, int n_images,
                        const vector<int> &dense_architecture = {64, 32});

std::unordered_map<std::string, std::string> loadEnvFile(const std::string &filename);

void showFilter(const string &layer_name,
                const vector<vector<MatrixXd>> &filters,
                int cell_size,
                int padding);
void showFilterIndividualOutputs(const string &layer_name, const vector<vector<MatrixXd>> &filters, int cell_size);
void showFilterEnhanced(const string &timeStamp, const string &layer_name, const vector<vector<MatrixXd>> &filters, bool use_grayscale, int colormap_type = cv::COLORMAP_JET, bool normalize_per_filter = true, bool show_values = false , int cell_size = 50, int padding = 3);
#endif