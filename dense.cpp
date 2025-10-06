#include "dense.hpp"

// int main() {

//   MatrixXd inputs = generateSyntheticData(2, 10, 42);
//   DenseLayer layer1(10, 5);
//   DenseLayer layer2(5, 3);
  
//   MatrixXd layer1_outputs = layer1.forward(inputs);
//   MatrixXd layer2_outputs = layer2.forward(layer1_outputs);

//   std::cout << std::endl;
//   std::cout << ">>>\n" << layer1_outputs << std::endl;
//   std::cout << ">>>\n" << layer2_outputs << std::endl;
//   std::cout << std::endl;


//   return 0;
// }

// Generate synthetic dataset with optional seed for reproducibility
MatrixXd generateSyntheticData(int samples, int features, unsigned seed = std::random_device{}()) {
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    MatrixXd data(samples, features);
    for (int i = 0; i < samples; ++i) {
        for (int j = 0; j < features; ++j) {
            data(i, j) = distribution(generator);
        }
    }
    return data;
}
// // Generate 100 samples with 5 features
// MatrixXd syntheticData = generateSyntheticData(100, 5);

// // Generate reproducible data with fixed seed
// MatrixXd reproducibleData = generateSyntheticData(100, 5, 42);

DenseLayer::DenseLayer(int n_inputs, int n_neurons)
{
    weights = MatrixXd::Random(n_inputs,n_neurons);
    biases = MatrixXd::Zero(1, n_neurons);
}

  MatrixXd& DenseLayer::forward(const MatrixXd& inputs) {
        // inputs shape: (batch_size, n_inputs)
        // outputs shape: (batch_size, n_neurons)
        outputs = inputs * weights + biases.replicate(inputs.rows(), 1);
        return outputs;
    }
    
const MatrixXd& DenseLayer::getOutputs(){ return outputs; }