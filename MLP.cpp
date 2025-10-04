#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int N = 3, M = 4, O = 3, P = 3;

MatrixXd inputs(O,M);

MatrixXd weights(N, M);
RowVectorXd biases(N);
MatrixXd layer1_outputs(O, N);

MatrixXd weights2(P, N);
RowVectorXd biases2(P);
MatrixXd layer2_outputs(O, P);


int main() {

  inputs << 1.0, 2.0, 3.0, 2.5, 
            2.0, 5.0, -1.0, 2.0,
            -1.5, 2.7, 3.3, -0.8;
  weights << 0.2, 0.8, -0.5, 1.0,
             0.5, -0.91, 0.26, -0.5,
             -0.26, -0.27, 0.17, 0.87;
  biases << 2.0, 3.0, 0.5;
  weights2 << 0.1, -0.14, 0.5,
              -0.5, 0.12, -0.33,
              -0.44, 0.73, -0.13;
  biases2 << -1.0, 2.0, -0.5;

  layer1_outputs = (inputs * weights.transpose());
  layer1_outputs.rowwise() += biases;

  layer2_outputs = (layer1_outputs * weights2.transpose());
  layer2_outputs.rowwise() += biases2;

  std::cout << std::endl;
  std::cout << ">>>\n" << layer1_outputs << std::endl;
  std::cout << ">>>\n" << layer2_outputs << std::endl;
  std::cout << std::endl;


  return 0;
}

