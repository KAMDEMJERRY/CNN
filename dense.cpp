#include "dense.hpp"






// int main() {
// //     Activation_ReLU();

// //   MatrixXd inputs = generateSyntheticData(2, 10, 42);

// //   DenseLayer layer1(10, 5);
// //   DenseLayer layer2(5, 3);
  
// //   MatrixXd layer1_outputs = layer1.forward(inputs);
// //   MatrixXd layer2_outputs = layer2.forward(layer1_outputs);

// //   std::cout << std::endl;
// //   std::cout << ">>>\n" << layer1_outputs << std::endl;
// //   std::cout << ">>>\n" << layer2_outputs << std::endl;
// //   std::cout << std::endl;


// //   VectorXd layer_outputs;
// //   layer_outputs = {4.8, 1.21, 2.385};

 
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
        output = inputs * weights + biases.replicate(inputs.rows(), 1);
        return output;
}
    
const MatrixXd& DenseLayer::getOutput(){ return output; }

MatrixXd& Activation_ReLU::forward(const MatrixXd& inputs){
    output = inputs.array().max(0);
    return output;
}

void testActivation_ReLU(){
    
    // Create dataset
    MatrixXd X = generateSyntheticData(1, 2);

    // Create Dense Layer with 2 input features and 3 output values
    DenseLayer dense1(2, 3);

    // Create ReLU activation(to be used with Dense Layer)
    Activation_ReLU activation1;
    
    // Make a forward pass of out training data through this layer
    dense1.forward(X);

    // Make a forward pass of our training data through this layer
    activation1.forward(dense1.output);

    cout << "Dense" << endl;
    cout << dense1.output << endl;
    cout << "ReLU" << endl;
    cout << activation1.output.topRows(1);
    cout << endl << endl;
}

MatrixXd& Activation_Softmax::forward(const MatrixXd& inputs){
  // Convertir en Array pour les opérations élément par élément
    ArrayXXd X_array = inputs.array();
    
    // Soustraire le max de chaque ligne
    ArrayXd max_vals = X_array.rowwise().maxCoeff();
    ArrayXXd shifted_X = X_array.colwise() - max_vals;
    
    // Exponentielle et normalisation
    ArrayXXd exp_X = shifted_X.exp();
    ArrayXd sums = exp_X.rowwise().sum();
    ArrayXXd result = exp_X.colwise() / sums;
    output = result.matrix();
    return output;
}