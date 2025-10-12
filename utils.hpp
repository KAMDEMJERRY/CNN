#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>
#include <Eigen/Dense>
#include <random>
using namespace std;
using namespace Eigen;
MatrixXd one_hot(VectorXd& y, int num_labels=0);
#endif 