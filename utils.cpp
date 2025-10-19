#include "utils.hpp"

MatrixXd one_hot(const VectorXd& y, int num_labels){
    int uniq = 0;
    int n_samples = y.size();
    if(num_labels<=0){
        uniq =  y.maxCoeff() + 1;
    }else{
        uniq = num_labels;
    }

    MatrixXd ary = MatrixXd::Zero(n_samples, uniq);
    for(int i=0; i<y.size(); i++){
        ary(i, static_cast<int>(y(i))) = 1;
    }
    return ary;
}


void logCNNArchitecture(const ImageDataset& imgDataset, 
                              const ConvLayer& conv1, const PoolLayer& pool1,
                              const ConvLayer& conv2, const PoolLayer& pool2,
                              int image_size, int input_channels, int n_images,
                              const vector<int>& dense_architecture ) {
    
    cout << "\n=== ARCHITECTURE COMPLÈTE DU CNN ===" << endl;
    cout << "======================================" << endl;

    int flattened_size = pool2.output_size * pool2.output_size * pool2.input_ch;
    
    // Construction de l'architecture dense dynamiquement
    vector<int> full_architecture;
    full_architecture.push_back(flattened_size);
    full_architecture.insert(full_architecture.end(), dense_architecture.begin(), dense_architecture.end());
    full_architecture.push_back(imgDataset.classes.size());

    // Partie convolutionnelle
    cout << "\n--- PARTIE CONVOLUTIONNELLE ---" << endl;
    cout << "Input: " << image_size << "x" << image_size << "x" << input_channels << endl;
    
    vector<pair<string, string>> conv_layers = {
        {"Conv1", "(" + to_string(conv1.filter_size) + "x" + to_string(conv1.filter_size) + 
                  ", filters=" + to_string(conv1.output_ch) + ")"},
        {"Pool1", "(pool_size=" + to_string(pool1.pool_size) + ")"},
        {"Conv2", "(" + to_string(conv2.filter_size) + "x" + to_string(conv2.filter_size) + 
                  ", filters=" + to_string(conv2.output_ch) + ")"},
        {"Pool2", "(pool_size=" + to_string(pool2.pool_size) + ")"}
    };
    
    vector<tuple<int, int, int>> dimensions = {
        {image_size, image_size, input_channels},
        {conv1.output_size, conv1.output_size, conv1.output_ch},
        {pool1.output_size, pool1.output_size, pool1.input_ch},
        {conv2.output_size, conv2.output_size, conv2.output_ch},
        {pool2.output_size, pool2.output_size, pool2.input_ch}
    };
    
    for (size_t i = 0; i < conv_layers.size(); ++i) {
        auto [h, w, c] = dimensions[i];
        auto [name, info] = conv_layers[i];
        auto [h_next, w_next, c_next] = dimensions[i+1];
        
        cout << (i == 0 ? "┌─ " : "├─ ") << name << ": " 
             << h << "x" << w << "x" << c << " → " 
             << h_next << "x" << w_next << "x" << c_next 
             << " " << info << endl;
    }
    cout << "└─ Flatten: → " << flattened_size << " features" << endl;

    // Partie dense
    cout << "\n--- PARTIE DENSE ---" << endl;
    int total_dense_params = 0;
    for (size_t i = 0; i < full_architecture.size() - 1; ++i) {
        int input_size = full_architecture[i];
        int output_size = full_architecture[i+1];
        int layer_params = input_size * output_size + output_size;
        total_dense_params += layer_params;
        
        string layer_name = (i == full_architecture.size() - 2) ? "Output" : 
                           "Dense" + to_string(i+1);
        string activation = (i == full_architecture.size() - 2) ? "Softmax" : "ReLU";
        
        cout << (i == 0 ? "┌─ " : "├─ ") << layer_name << ": " 
             << input_size << " → " << output_size
             << " | params: " << layer_params 
             << " → " << activation << endl;
    }

    // Résumé
    cout << "\n--- RÉSUMÉ ---" << endl;
    cout << "Architecture: ";
    for (size_t i = 0; i < full_architecture.size(); ++i) {
        cout << full_architecture[i];
        if (i < full_architecture.size() - 1) cout << " → ";
    }
    cout << endl;
    
    cout << "Total paramètres: " << total_dense_params << " (dense only)" << endl;
    cout << "Taille input: " << n_images << " images " << image_size << "x" << image_size << endl;
    cout << "Taille output: " << n_images << " × " << imgDataset.classes.size() << " probabilités" << endl;
}




void displayPredictions(const MatrixXd& predictions, 
                       const vector<int>& true_labels, 
                       const VectorXd& Y_encoded,
                       int num_samples) {
    cout << "\n=== PRÉDICTIONS (" << num_samples << " premières images) ===" << endl;
    
    int n_display = min(num_samples, static_cast<int>(true_labels.size()));
    
    for (int i = 0; i < n_display; ++i) {
        cout << "Image " << i + 1 << " - Label réel: " << true_labels[i] 
             << " (" << Y_encoded[i] << ")" << endl;
        
        cout << "Probabilités: [";
        for (int j = 0; j < predictions.cols(); ++j) {
            cout << predictions(i, j);
            if (j < predictions.cols() - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Trouver la classe prédite
        int predicted_class = 0;
        double max_prob = predictions(i, 0);
        for (int j = 1; j < predictions.cols(); ++j) {
            if (predictions(i, j) > max_prob) {
                max_prob = predictions(i, j);
                predicted_class = j;
            }
        }
        cout << "Classe prédite: " << predicted_class 
             << " (prob: " << max_prob << ")" << endl << endl;
    }
}

// int main(int argc, char**argv){
    
//     VectorXd y(6);
//     y << 0, 1, 3, 3, 2, 1; 
//     MatrixXd y_enc = one_hot(y);
//     cout << "y encoded >>> \n" << y_enc << endl;
//     return 0;
// }