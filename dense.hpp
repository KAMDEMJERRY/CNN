#ifndef DENSE_HPP
#define DENSE_HPP

#include <iostream>
#include <Eigen/Dense>
#include <random>

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

    DenseLayer(int n_inputs, int n_neurons);
    MatrixXd& forward(const MatrixXd& inputs);
    const MatrixXd& getOutput();

};
class Activation_ReLU{
public:

    MatrixXd output;

    MatrixXd& forward(const MatrixXd& inputs);
};

class Activation_Softmax{
public:
    MatrixXd output;
    MatrixXd& forward(const MatrixXd& inputs);
};
extern MatrixXd generateSyntheticData(int samples, int features, unsigned seed);
 
extern void testActivation_ReLU();

#endif // DENSE_HPP
