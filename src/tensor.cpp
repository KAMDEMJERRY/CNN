#include <unsupported/Eigen/CXX11/Tensor>
#include "matplotlibcpp.h"
#include <cmath>
#include <iostream>
#include <string>
#include "utils.hpp"

namespace plt = matplotlibcpp;
using namespace std;
using namespace Eigen;

void test_tensor_convolution()

{
    Eigen::Tensor<float, 4> input(1, 6, 6, 3);
    input.setRandom();

    Eigen::Tensor<float, 3> kernel(2, 3, 3);
    kernel.setRandom();

    Eigen::Tensor<float, 4> output(1, 4, 4, 3);

    Eigen::array<int, 2> dims({1, 2});
    output = input.convolve(kernel, dims);

    std::cout << "input:\n\n"
              << input << "\n\n";
    std::cout << "kernel:\n\n"
              << kernel << "\n\n";
    std::cout << "output:\n\n"
              << output << "\n\n";
}

void test_matplolibcpp()

{
    // Prepare data.
    int n = 5000;
    std::vector<double> x(n), y(n), z(n), w(n, 2);
    for (int i = 0; i < n; ++i)
    {
        x.at(i) = i * i;
        y.at(i) = sin(2 * M_PI * i / 360.0);
        z.at(i) = log(i);
    }

    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);

    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(x, y);

    // Plot a red dashed line from given x and y data.
    plt::plot(x, w, "r--");

    // Plot a line whose name will show up as "log(x)" in the legend.
    plt::plot(x, z, {{"label", "log(x)"}});

    // Set x-axis to interval [0,1000000]
    plt::xlim(0, 1000 * 1000);

    // Add graph title
    plt::title("Sample figure");

    // Enable legend.
    plt::legend();
    plt::show();

    // Save the image (file format is determined by the extension)
    // plt::save("./basic.png");
}

class NeuralNetwork
{
public:
    NeuralNetwork(const std::vector<int> &layers_sizes)
    {

        std::cout << "Neural Network Initialized!" << std::endl;
    }

private:
    // std::vector<std::unique_ptr<Layer>> layers;
};

int main(int argc, char **argv)

{
    MatrixXd img1 = MatrixXd::Random(20, 20);
    MatrixXd img2 = MatrixXd::Random(20, 20);
    MatrixXd img3 = MatrixXd::Random(20, 20);

    vector<vector<MatrixXd>> filters = {{img1},
                                        {img2},
                                        {img3}};

    showFilterEnhanced("Tensor", "convlayer", filters, false, cv::COLORMAP_JET, true, true, 50, 3);


    return 0;
}
