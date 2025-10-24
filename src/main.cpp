#include "convolution.hpp"
#include "dense.hpp"
#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <utility>

using namespace std;
using namespace Eigen;



int main() {
    try {




        // Charger le dataset d'images
        cout << "=== CHARGEMENT DU DATASET ===" << endl;
        ImageDataset imgDataset = loadDataSet();
        int n_images = imgDataset.images.size();
        vector<int> Y = imgDataset.getY_encoded();
        VectorXd y(Y.size());
        for(int i = 0; i < Y.size(); i++) {
            y(i) = static_cast<double>(Y[i]);
        }

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
        
        // Activation_Softmax activation3;

        // LossCategoricalCrossentropy loss_function;
        Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;

        // Log de l'architecture;
        logCNNArchitecture(imgDataset, conv1, pool1, conv2, pool2, image_size, input_channels, n_images);

        Optimizer_SGD optimizer(.02);



     










        std::vector<std::vector<MatrixXd>> inputs(n_images);
        for (int img_idx = 0; img_idx < n_images; ++img_idx) {
            inputs[img_idx].push_back(imgDataset.images[img_idx]);
        }    
       




        // === CONVOLUTION SUR TOUTES LES IMAGES ===
        // cout << "\n=== PHASE DE CONVOLUTION ===" << endl;

        
        
        






        // === CLASSIFICATION ===

        for(int i_ = 0; i_ < 100; i_++){
    

            // cout << "\n=== PHASE DE CONVOLUTION ===" << endl;
            // Forward pass through convolutional layers
            conv1.forward(inputs);
            pool1.forward(conv1.output_maps);
            conv2.forward(pool1.output_maps);
            pool2.forward(conv2.output_maps);                
            X = pool2.flatten();
        
            // cout << "\n=== PHASE DE CLASSIFICATION ===" << endl;
            dense1.forward(X);
            activation1.forward(dense1.output);
            dense2.forward(activation1.output);
            activation2.forward(dense2.output);
            dense3.forward(activation2.output);
            double loss = loss_activation.forward(static_cast<const MatrixXd&>(dense3.output), 
                                     static_cast<const VectorXd&>(y));


            // Calcul de la précision
            int correct_predictions = 0;
            for (int i = 0; i < n_images; ++i) {
                int predicted_class = 0;
                double max_prob = loss_activation.output(i, 0);
                for (int j = 1; j < loss_activation.output.cols(); ++j) {
                    if (loss_activation.output(i, j) > max_prob) {
                        max_prob = loss_activation.output(i, j);
                        predicted_class = j;
                    }
                }
                if (predicted_class == Y[i]) {
                    correct_predictions++;
                }
            }
            double accuracy = static_cast<double>(correct_predictions) / n_images * 100.0;
            
            cout << "=== RÉSULTATS ===" << endl;
            cout << "mise a jour iteration: " << i_;
            cout << "  loss : " << loss;
            cout << "  acc: " << accuracy << "% (" << correct_predictions << "/" << n_images << ")" << endl;
            
            loss_activation.backward(loss_activation.output, y);
            dense3.backward(loss_activation.dinputs);
            activation2.backward(dense3.dinputs);
            
            dense2.backward(activation2.dinputs);


            activation1.backward(dense2.dinputs);
            dense1.backward(activation1.dinputs);
            pool2.backward(pool2.unflatten(dense1.dinputs));
            conv2.backward(pool2.dinput);
            pool1.backward(conv2.dinputs);
            conv1.backward(pool1.dinput);

            optimizer.update_params(dense1);
            optimizer.update_params(dense2);
            optimizer.update_params(dense3);
            optimizer.update_params(conv1);
            optimizer.update_params(conv2);

            
        }
    
    } catch (const std::exception& e) {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n=== PROCESSUS TERMINÉ AVEC SUCCÈS ===" << endl;
    return 0;
}