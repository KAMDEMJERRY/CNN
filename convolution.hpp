#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include "imgdataset.hpp"

using namespace Eigen;
using namespace std;

// Déclaration de la classe ConvLayer
class ConvLayer {
public: 
    int input_size;
    int input_ch;
    int filter_size;
    int output_ch;
    int padding;
    int stride;
    int output_size;
   
    std::vector<std::vector<MatrixXd>> filters;
    VectorXd biases;
    std::vector<std::vector<MatrixXd>> output_maps;

    // Constructeur
    ConvLayer(int in_size, int in_ch, int f_num, int f_size, int pad = 1, int str = 1);

    // Méthodes
    void initialize();
    void forward(const std::vector<std::vector<MatrixXd>>& batch_input_maps);
};

// Déclaration de la classe PoolLayer
class PoolLayer {
public:
    int input_size;
    int input_ch;
    int pool_size;
    int output_size;

    std::vector<std::vector<MatrixXd>> input_maps;
    std::vector<std::vector<MatrixXd>> output_maps;
    std::vector<std::vector<MatrixXd>> dvalue;

    MatrixXd flats_output;
    
    
    // Constructeur
    PoolLayer(int in_size, int in_ch, int p_size);

    // Méthodes
    void forward(const std::vector<std::vector<MatrixXd>>& batch_in_maps);
    vector<vector<MatrixXd>> &unflatten(MatrixXd &flats);
    MatrixXd &flatten();
};


#endif // CONVOLUTION_HPP