#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <stdexcept>

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

    std::vector<std::vector<MatrixXd>> inputs;
    std::vector<std::vector<MatrixXd>> dinputs;
    std::vector<std::vector<MatrixXd>> dweights;
    VectorXd dbiases;

    // Constructeur
    ConvLayer(int in_size, int in_ch, int f_num, int f_size, int pad = 1, int str = 1);
    ConvLayer() {};
    // Méthodes
    void initialize();
    void forward(const std::vector<std::vector<MatrixXd>>& batch_input_maps);
    std::vector<std::vector<MatrixXd>> &backward(const std::vector<std::vector<MatrixXd>> &dvalue);
};


class Activation_ReLU_Conv{
public:
    std::vector<std::vector<MatrixXd>> inputs;
    std::vector<std::vector<MatrixXd>> outputs;
    std::vector<std::vector<MatrixXd>> dinputs;
    
    std::vector<std::vector<MatrixXd>>& forward(const std::vector<std::vector<MatrixXd>>& inputs);
    std::vector<std::vector<MatrixXd>>& backward(const std::vector<std::vector<MatrixXd>>& dvalues);
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
    std::vector<std::vector<MatrixXd>> dinput;
    std::vector<std::vector<std::vector<std::pair<int, int>>>> max_indices;

    MatrixXd flats_output;
    
    
    // Constructeur
    PoolLayer(int in_size, int in_ch, int p_size);
    PoolLayer(){};

    // Méthodes
    void forward(const std::vector<std::vector<MatrixXd>>& batch_in_maps);
    vector<vector<MatrixXd>> &unflatten(MatrixXd &flats);
    MatrixXd &flatten();
    vector<vector<MatrixXd>> &backward(std::vector<std::vector<MatrixXd>>& dvalue);
};


#endif // CONVOLUTION_HPP

