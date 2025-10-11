#include "convolution.hpp"
#include "dense.hpp"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace Eigen;
void logCNNArchitecture(const ImageDataset& imgDataset, 
                              const ConvLayer& conv1, const PoolLayer& pool1,
                              const ConvLayer& conv2, const PoolLayer& pool2,
                              int image_size, int input_channels, int n_images,
                              const vector<int>& dense_architecture = {64, 32}) {
    
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
int main() {
    try {
        // Charger le dataset d'images
        cout << "=== CHARGEMENT DU DATASET ===" << endl;
        ImageDataset imgDataset = loadDataSet();
        int n_images = imgDataset.images.size();
        vector<int> Y = imgDataset.getY_encoded();

        // Vérifier que des images ont été chargées
        if (imgDataset.images.empty()) {
            throw std::runtime_error("Aucune image chargée dans le dataset");
        }
        
        // Afficher les informations du dataset
        cout << "Nombre d'images chargées: " << imgDataset.images.size() << endl;
        cout << "Dimensions des images: " << imgDataset.images[0].rows() << "x" << imgDataset.images[0].cols() << endl;
        cout << "Nombre de classes: " << imgDataset.classes.size() << endl;
        cout << "Classes: ";
        for (const auto& cls : imgDataset.classes) {
            cout << cls << " ";
        }
        cout << "\n\n";

        // Prendre la première image comme exemple
        MatrixXd first_image = imgDataset.images[0];
        cout << "Première image (extrait 10x10):\n" << first_image.block(0, 0, 10, 10) << "\n\n";
        cout << "Label de la première image: " << imgDataset.labels[0] << "\n\n";

        int image_size = first_image.rows(); // Les images sont carrées (128x128)
        int input_channels = 1; // Images en niveaux de gris


        
        // Création de l'architecture CNN
        cout << "=== CONFIGURATION DU CNN ===" << endl;
      
        // Première couche de convolution
        ConvLayer conv1(image_size, input_channels, 8, 3, 1, 1); // 8 filtres de 3x3
     
        // Première couche de pooling
        PoolLayer pool1(conv1.output_size, conv1.output_ch, 2); // Pooling 2x2
        
        // Deuxième couche de convolution
        ConvLayer conv2(pool1.output_size, pool1.input_ch, 16, 3, 1, 1); // 16 filtres de 3x3
        
        // Deuxième couche de pooling
        PoolLayer pool2(conv2.output_size, conv2.output_ch, 2); // Pooling 2x2

        // Calcul de la taille après aplatissement
        int flattened_size = pool2.output_size * pool2.output_size * pool2.input_ch;
        
        // Préparer la matrice d'entrée pour les couches denses
        MatrixXd X(n_images, flattened_size);

        // Initialiser les couches denses
        DenseLayer dense1(flattened_size, 64);  // Augmenté la taille pour plus de capacité
       
        Activation_ReLU activation1;

        DenseLayer dense2(64, 32);              // Couche intermédiaire
        
        Activation_ReLU activation2;

        DenseLayer dense3(32, imgDataset.classes.size());  // Sortie = nombre de classes
        
        Activation_Softmax activation3;

        // Calcul du total des paramètres
        int total_params = (flattened_size * 64 + 64) + (64 * 32 + 32) + (32 * imgDataset.classes.size() + imgDataset.classes.size());
        

        logCNNArchitecture(imgDataset, conv1, pool1, conv2, pool2, image_size, input_channels, n_images);

        // === CONVOLUTION SUR TOUTES LES IMAGES ===
        cout << "\n=== PHASE DE CONVOLUTION ===" << endl;
        
        for (int img_idx = 0; img_idx < n_images; ++img_idx) {
            if (img_idx % 100 == 0) {
                cout << "Traitement de l'image " << img_idx + 1 << "/" << n_images << endl;
            }
            
            // Préparer l'input
            std::vector<MatrixXd> current_input;
            current_input.push_back(imgDataset.images[img_idx]);
            
            // Forward pass through convolutional layers
            conv1.forward(current_input);
            pool1.forward(conv1.output_maps);
            conv2.forward(pool1.output_maps);
            pool2.forward(conv2.output_maps);
            
            // Aplatir la sortie et la stocker dans X
            pool2.flatten();
            X.row(img_idx) = pool2.flats_output;
        }

        cout << "Matrice X créée: " << X.rows() << " x " << X.cols() << endl;

        // === CLASSIFICATION ===
        cout << "\n=== PHASE DE CLASSIFICATION ===" << endl;
        

        // Forward pass through dense layers
        cout << "Forward pass through dense layers..." << endl;
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);
        activation3.forward(dense3.output);

        // Afficher les prédictions pour les premières images
        // cout << "\n=== PRÉDICTIONS (5 premières images) ===" << endl;
        int num_display = min(5, n_images);
        for (int i = 0; i < num_display; ++i) {
            cout << "Image " << i + 1 << " - Label réel: " << imgDataset.labels[i] << " (" << Y[i] << ")" << endl;
            cout << "Probabilités: [";
            for (int j = 0; j < activation3.output.cols(); ++j) {
                cout << activation3.output(i, j);
                if (j < activation3.output.cols() - 1) cout << ", ";
            }
            cout << "]" << endl;
            
            // Trouver la classe prédite
            int predicted_class = 0;
            double max_prob = activation3.output(i, 0);
            for (int j = 1; j < activation3.output.cols(); ++j) {
                if (activation3.output(i, j) > max_prob) {
                    max_prob = activation3.output(i, j);
                    predicted_class = j;
                }
            }
            cout << "Classe prédite: " << predicted_class << " (prob: " << max_prob << ")" << endl << endl;
        }

        // Calcul de la précision
        int correct_predictions = 0;
        for (int i = 0; i < n_images; ++i) {
            int predicted_class = 0;
            double max_prob = activation3.output(i, 0);
            for (int j = 1; j < activation3.output.cols(); ++j) {
                if (activation3.output(i, j) > max_prob) {
                    max_prob = activation3.output(i, j);
                    predicted_class = j;
                }
            }
            if (predicted_class == Y[i]) {
                correct_predictions++;
            }
        }
        
        double accuracy = static_cast<double>(correct_predictions) / n_images * 100.0;
        cout << "=== RÉSULTATS ===" << endl;
        cout << "Précision sur l'ensemble d'entraînement: " << accuracy << "% (" 
             << correct_predictions << "/" << n_images << ")" << endl;



    
    } catch (const std::exception& e) {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n=== PROCESSUS TERMINÉ AVEC SUCCÈS ===" << endl;
    return 0;
}