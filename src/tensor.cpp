#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <string>

// Exemple d'utilisation de Eigen::Tensor pour une convolution 2D basique
const int batch_size = 2;
const int channels = 2;
const int height = 3;
const int width = 4;
const int out_channels = 2;
const int in_channels = 2;
const int k_height = 2;
const int k_width = 2;

// Définition d'un tenseur 4D (Batch, Channels, Height, Width)
Eigen::Tensor<float, 3> input(batch_size, height, width);
// Eigen::Tensor<float, 4> kernel(out_channels, in_channels, k_height, k_width);

// Dimensions sur lesquelles appliquer la convolution (H et W)
// Eigen::array<ptrdiff_t, 2> dims({2, 3});

// Convolution (note : Eigen::Tensor::convolve effectue une convolution "valide" par défaut)
// Eigen::Tensor<float, 4> output = input.convolve(kernel, dims);

int main(int argc, char *argv[])
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
    return 0;
}