#ifndef DENSE_HPP
#define DENSE_HPP

#include <iostream>
#include <Eigen/Dense>
#include <random>
#include "utils.hpp"
#include "convolution.hpp"

using namespace std;
using namespace Eigen;

class DenseLayer{
public:
    int n_inputs;
    int n_neurons;
    MatrixXd inputs;
    MatrixXd weights;
    RowVectorXd biases;
    MatrixXd output;
    MatrixXd dinputs;
    MatrixXd dweights;
    MatrixXd dbiases;


    DenseLayer(int n_inputs, int n_neurons);
    MatrixXd& forward(const MatrixXd& inputs);
    void backward(const MatrixXd& dvalues);
    const MatrixXd& getOutput();

};
class Activation_ReLU{
public:

    MatrixXd output;
    MatrixXd dinputs;

    MatrixXd& forward(const MatrixXd& inputs);
    MatrixXd& backward(const MatrixXd& dvalues);
};

class Activation_Softmax{
public:
    MatrixXd output;
    MatrixXd dinputs;
    MatrixXd& forward(const MatrixXd& inputs);
    MatrixXd& backward(const MatrixXd& dvalues);
};

class LossCategoricalCrossentropy{
public:
    MatrixXd dinputs;
    VectorXd forward(const MatrixXd& y_pred, const MatrixXd& y) ;
    VectorXd forward(const MatrixXd& y_pred, const  VectorXd& y) ;
    double calculate(const MatrixXd& output, const MatrixXd& y) ;
    double calculate(const MatrixXd& output, const  VectorXd& y) ;
    MatrixXd& backward(const MatrixXd& dvalues, const  VectorXd& y);
    MatrixXd& backward(const MatrixXd& dvalues, const MatrixXd& y);
};
class Activation_Softmax_Loss_CategoricalCrossentropy{
public:
    Activation_Softmax activation;
    LossCategoricalCrossentropy loss;
    MatrixXd output;
    MatrixXd dinputs;

    double forward(const MatrixXd& inputs, const MatrixXd& y_true);
    double forward(const MatrixXd& inputs, const VectorXd& y_true);
    MatrixXd& backward(const MatrixXd& dvalues, const MatrixXd& y_true);
    MatrixXd& backward(const MatrixXd& dvalues, const VectorXd& y_true);

};

class Optimizer_SGD{
public:
    double learning_rate;
    Optimizer_SGD(double learning_rate=1.0);
    void update_params(DenseLayer& layer);
    void update_params(ConvLayer &layer);
};
extern MatrixXd generateSyntheticData(int samples, int features, unsigned seed);

extern void testActivation_ReLU();

#endif // DENSE_HPP
