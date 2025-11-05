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
    DenseLayer::n_inputs = n_inputs;
    DenseLayer::n_neurons = n_neurons;
    double scale = sqrt(2.0 / n_inputs);
    weights = MatrixXd::Random(n_inputs,n_neurons) * scale;
    biases = MatrixXd::Zero(1, n_neurons);
    dweights = MatrixXd::Zero(n_inputs, n_neurons);
    dbiases = MatrixXd::Zero(1, n_neurons);
    
    weights_momentum = MatrixXd::Zero(weights.rows(), weights.cols());
    biases_momentum = RowVectorXd::Zero(biases.size());
}

MatrixXd& DenseLayer::forward(const MatrixXd& inputs) {
        // inputs shape: (batch_size, n_inputs)
        // outputs shape: (batch_size, n_neurons)
        this->inputs = inputs;
        output = inputs * weights + biases.replicate(inputs.rows(), 1);
        return output;
}

void DenseLayer::backward(const MatrixXd& dvalues){
    try {
        // dvalues shape: (batch_size, n_neurons)
        
        // 1. Gradients des poids: inputs^T * dvalues
        // inputs: (batch_size, n_inputs) -> transpose: (n_inputs, batch_size)
        // dvalues: (batch_size, n_neurons)
        // dweights: (n_inputs, n_neurons)
        this->dweights = DenseLayer::inputs.transpose() * dvalues;
        
        // 2. Gradients des biais: sum sur le batch (garder forme ligne)
        // dvalues: (batch_size, n_neurons)
        // dbiases: (1, n_neurons)
        this->dbiases = dvalues.colwise().sum(); // Somme sur les colonnes
        
        // 3. Gradients des inputs: dvalues * weights^T
        // dvalues: (batch_size, n_neurons)
        // weights: (n_inputs, n_neurons) -> transpose: (n_neurons, n_inputs)
        // dinputs: (batch_size, n_inputs)
        DenseLayer::dinputs = dvalues * DenseLayer::weights.transpose();
        
    } catch(const std::exception& e) {
        std::string new_msg = std::string(e.what()) + " :: Dense Layer backward";
        std::cout << new_msg << std::endl;
        throw std::runtime_error(new_msg);
    }
}
   
const MatrixXd& DenseLayer::getOutput(){ return output; }

MatrixXd& Activation_ReLU::forward(const MatrixXd& inputs){
    
    this->inputs = inputs;
    output = inputs.array().max(0);
    return output;
}
MatrixXd& Activation_ReLU::backward(const MatrixXd& dvalues){
    try{
        Activation_ReLU::dinputs = dvalues.array() * (this->inputs.array() > 0).cast<double>();
        return Activation_ReLU::dinputs;
    }catch(const std::exception& e){
        std::string new_msg(std::string(e.what()) + " ::  Activation ReLU backward");
        std::cout << new_msg << std::endl;
        throw std::runtime_error(new_msg);
    }
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
MatrixXd& Activation_Softmax::backward(const MatrixXd& dvalues){
    int N = dvalues.rows();
    int M = dvalues.cols();
    dinputs.resize(N, M);
    for (int i = 0; i < N; i++) {
        // Utiliser VectorXd pour plus d'efficacité
        Eigen::VectorXd single_output = output.row(i).transpose();
        Eigen::VectorXd single_dvalues = dvalues.row(i).transpose();
        
        // Calculer la matrice jacobienne
        MatrixXd jacobian_matrix = MatrixXd(single_output.asDiagonal()) - (single_output * single_output.transpose());
        
        // Calculer le gradient
        dinputs.row(i) = (jacobian_matrix * single_dvalues).transpose();
    }
    
    return dinputs;
};

// 1: y one hot encoded         2: Vector of categories
VectorXd LossCategoricalCrossentropy::forward(const MatrixXd &y_pred, const MatrixXd& y){
    int samples = y_pred.rows();

    MatrixXd y_pred_clipped = y_pred.array().min(1e-7).max(1 - 1e-7).matrix();

    RowVectorXd correct_confidences(samples);
    correct_confidences = (y_pred.array() * y.array()).rowwise().sum();


   RowVectorXd neg_log_likelihoods = (correct_confidences.array().log()) * -1;
   return neg_log_likelihoods;
}
VectorXd LossCategoricalCrossentropy::forward(const MatrixXd& y_pred, const VectorXd& y){
    int samples = y_pred.rows();

    MatrixXd y_pred_clipped = y_pred.array().min(1e-7).max(1 - 1e-7).matrix();
    
    RowVectorXd correct_confidences(samples);
    for(int i = 0; i < y.size(); i++){
        correct_confidences(i) = y_pred.coeff(i, y(i));
    }

   RowVectorXd neg_log_likelihoods = (correct_confidences.array().log()) * -1;
   return neg_log_likelihoods;
}

double LossCategoricalCrossentropy::calculate(const MatrixXd& output, const MatrixXd& y){
    VectorXd sample_loss = forward(output, y);
    return sample_loss.mean();
}
double LossCategoricalCrossentropy::calculate(const MatrixXd& output, const VectorXd& y){
    VectorXd sample_loss = forward(output, y);
    return sample_loss.mean();
}

MatrixXd& LossCategoricalCrossentropy::backward(const MatrixXd& dvalues, const VectorXd& y_true){
    // Nombre d'observations
    int samples = dvalues.rows();
    // Nombre de classes
    int labels = dvalues.row(0).cols();
    // Si les labels sont fins, les encoder en one hot
    MatrixXd y_true_onehot = one_hot(y_true, labels);
    
    LossCategoricalCrossentropy::dinputs = -y_true_onehot.array() / dvalues.array();
    LossCategoricalCrossentropy::dinputs = LossCategoricalCrossentropy::dinputs.array() / samples;
    
    return LossCategoricalCrossentropy::dinputs;
}   
MatrixXd& LossCategoricalCrossentropy::backward(const MatrixXd& dvalues, const MatrixXd& y_true){
    // Nombre d'observations
    int samples = dvalues.rows();
    // Nombre de classes
    int labels = dvalues.row(0).cols();
    
    LossCategoricalCrossentropy::dinputs = -y_true.array() / dvalues.array();
    LossCategoricalCrossentropy::dinputs = LossCategoricalCrossentropy::dinputs.array() / samples;
    
    return LossCategoricalCrossentropy::dinputs;
} 


double Activation_Softmax_Loss_CategoricalCrossentropy::forward(const MatrixXd& inputs, const MatrixXd& y_true) {
    activation.forward(inputs);
    output = activation.output;
    return loss.calculate(output, y_true);
}

double Activation_Softmax_Loss_CategoricalCrossentropy::forward(const MatrixXd& inputs, const VectorXd& y_true) {
    activation.forward(inputs);
    output = activation.output;
    return loss.calculate(output, y_true);
}

// Backward pass optimisé
MatrixXd& Activation_Softmax_Loss_CategoricalCrossentropy::backward(const MatrixXd& dvalues, const MatrixXd& y_true) {
    int samples = dvalues.rows();
    int classes = dvalues.cols();
    
    VectorXi y_true_discrete(samples);
    
    // Conversion one-hot vers discret
    if (y_true.cols() > 1) {
        // Utiliser Eigen pour trouver les indices max par ligne
        for (int i = 0; i < samples; i++) {
            int max_index;
            y_true.row(i).maxCoeff(&max_index);
            y_true_discrete(i) = max_index;
        }
    } else {
        y_true_discrete = y_true.cast<int>();
    }

    dinputs = dvalues;
    
    // Appliquer le gradient
    for (int i = 0; i < samples; i++) {
        dinputs(i, y_true_discrete(i)) -= 1.0;
    }
    
    dinputs /= samples;
    return dinputs;
}

MatrixXd& Activation_Softmax_Loss_CategoricalCrossentropy::backward(const MatrixXd& dvalues, const VectorXd& y_true) {
    try{
        int samples = dvalues.rows();
        
        dinputs = dvalues;
        
        VectorXi y_true_int = y_true.cast<int>();
        for (int i = 0; i < samples; i++) {
            dinputs(i, y_true_int(i)) -= 1.0;
        }
        
        dinputs = dinputs.array() / samples;
        return dinputs;

    }catch(const std::exception& e){
        std::string new_msg(std::string(e.what()) + " :: Activation Softmax Loss Categorical backward\n");
        std::cout << new_msg << std::endl;
        throw std::runtime_error(new_msg);
    }

}


Optimizer_SGD::Optimizer_SGD(double learning_rate, double decay, double momentum):learning_rate(learning_rate), current_learning_rate(learning_rate), decay(decay){
    iterations = 0;
    this->momentum = momentum;

}

void Optimizer_SGD::pre_update_params()
{
    if(decay){
        current_learning_rate = learning_rate * (1. / (1. + decay * iterations));
    }
}

void Optimizer_SGD::update_params(DenseLayer &layer)
{
    MatrixXd weights_updates;
    RowVectorXd biases_updates;
    if(momentum){
        weights_updates = momentum * layer.weights_momentum - current_learning_rate * layer.dweights;
        layer.weights_momentum = weights_updates;
        
        biases_updates = momentum * layer.biases_momentum - current_learning_rate * layer.dbiases;
        layer.biases_momentum = biases_updates;
    }
    else{
        weights_updates = - current_learning_rate * layer.dweights.array();
        biases_updates = - current_learning_rate * layer.dbiases.array();
    }

    layer.weights = layer.weights.array() + weights_updates.array();
    layer.biases = layer.biases.array() + biases_updates.array();

}

void Optimizer_SGD::update_params(ConvLayer& layer){
    // std::ofstream debug_file("/home/ndomboukamdem/Documents/INFL/Master 2/Code/log/optimizer_debug.txt", std::ios::app);

    std::vector<std::vector<MatrixXd>> filters_updates(layer.output_ch, std::vector<MatrixXd>(layer.input_ch));
    VectorXd biases_updates(layer.output_ch); 
    
    // Initialiser les matrices de mise à jour
    for(int oc = 0; oc < layer.output_ch; ++oc) {
        for(int ic = 0; ic < layer.input_ch; ++ic) {
            filters_updates[oc][ic] = MatrixXd::Zero(layer.filter_size, layer.filter_size);
            // debug_file << "weights gradient grad[" << oc << "][" << ic << "] to zeros.\n" << layer.dweights[oc][ic] << std::endl;
        }
    }
    biases_updates.setZero();
    
    if(momentum){
        // Calculer les updates avec momentum
        for(int oc = 0; oc < layer.output_ch; oc++){
            for(int ic = 0; ic < layer.input_ch; ic++){
                filters_updates[oc][ic] = (momentum * layer.filters_momentum[oc][ic].array()).array() - (current_learning_rate * layer.dweights[oc][ic].array()).array();
                layer.filters_momentum[oc][ic] = filters_updates[oc][ic];
            }
        }

        biases_updates = momentum * layer.biases_momentum.array() - current_learning_rate * layer.dbiases.array();
        layer.biases_momentum = biases_updates;
    } else {
        // Calculer les updates sans momentum
        for(int oc = 0; oc < layer.output_ch; oc++){
            for(int ic = 0; ic < layer.input_ch; ic++){
                filters_updates[oc][ic] = -current_learning_rate * layer.dweights[oc][ic].array();
            }
        }
        biases_updates = -current_learning_rate * layer.dbiases.array();
    }

    // Appliquer les updates aux paramètres
    for(int oc = 0; oc < layer.output_ch; oc++){
        for(int ic = 0; ic < layer.input_ch; ic++){
            layer.filters[oc][ic] = layer.filters[oc][ic].array() + filters_updates[oc][ic].array();
        }
    }
    layer.biases = layer.biases.array() + biases_updates.array();
}

void Optimizer_SGD::post_update_params(){
    iterations += 1;
}
