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
    MatrixXd outputs;

    DenseLayer(int n_inputs, int n_neurons);
    MatrixXd& forward(const MatrixXd& inputs);
    const MatrixXd& getOutputs();

};

extern MatrixXd generateSyntheticData(int samples, int features, unsigned seed);
 


#endif // DENSE_HPP
