#include <iostream>
#include <Eigen/Dense>
#include <fstream>

int flat[9] = {0};
Eigen::MatrixXd input(1, 9);

// Corrected dimensions - biases should be row vectors
Eigen::MatrixXd layer1(9, 2);
Eigen::RowVectorXd layer1_b(2);  // Changed to RowVectorXd
Eigen::MatrixXd layer1_out(1, 2);  // Fixed: should be (1, 2) not (1, 9)

Eigen::MatrixXd layer2(2, 3);
Eigen::RowVectorXd layer2_b(3);   // Changed to RowVectorXd
Eigen::MatrixXd layer2_out(1, 3);

Eigen::MatrixXd layer3(3, 2);
Eigen::RowVectorXd layer3_b(2);   // Changed to RowVectorXd
Eigen::MatrixXd layer3_out(1, 2);

Eigen::MatrixXd relu(Eigen::MatrixXd x) {
    return x.cwiseMax(0);
}

void lireFichier() {
    std::ifstream file("conv_output.txt");
    if(file.is_open()) {
        for(int i = 0; i < 9; i++) {
            file >> flat[i];
        }
        file.close();
    }
}

void flat_to_input() {
    for(int i = 0; i < 9; i++) {
        input(0, i) = flat[i];
    }
}

int main() {
    lireFichier();
    flat_to_input();

    // Initialize weights and biases (important!)
    layer1 = Eigen::MatrixXd::Random(9, 2);
    layer1_b = Eigen::RowVectorXd::Random(2);
    
    layer2 = Eigen::MatrixXd::Random(2, 3);
    layer2_b = Eigen::RowVectorXd::Random(3);
    
    layer3 = Eigen::MatrixXd::Random(3, 2);
    layer3_b = Eigen::RowVectorXd::Random(2);

    std::cout << "Input: \n" << input << std::endl;

    layer1_out = relu((input * layer1).rowwise() + layer1_b);
    std::cout << "Layer 1 output: \n" << layer1_out << std::endl;
    
    layer2_out = relu((layer1_out * layer2).rowwise() + layer2_b);
    std::cout << "Layer 2 output: \n" << layer2_out << std::endl;   

    layer3_out = relu((layer2_out * layer3).rowwise() + layer3_b);
    std::cout << "Layer 3 output: \n" << layer3_out << std::endl;

    return 0;
}