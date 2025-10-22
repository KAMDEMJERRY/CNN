// g++ -std=c++17 -I. -I/usr/include/eigen3 test_.cpp dense.cpp utils.cpp  -o dense_demo `pkg-config --cflags --libs opencv4`
#include "dense.hpp"
#include "utils.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>

using namespace std;
using namespace Eigen;

/**
 * @brief Génère des données d'entraînement fictives
 * 
 * @param n_samples Nombre d'échantillons
 * @param n_features Nombre de caractéristiques
 * @param n_classes Nombre de classes
 * @return pair<MatrixXd, VectorXd> Données X et labels y
 */
pair<MatrixXd, VectorXd> generate_fake_data(int n_samples, int n_features, int n_classes) {
    // Générateur de nombres aléatoires
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-1.0, 1.0);
    uniform_int_distribution<int> class_dist(0, n_classes - 1);
    
    // Générer les données d'entrée
    MatrixXd X(n_samples, n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = dist(gen);
        }
    }
    
    // Générer les labels (classes cibles)
    VectorXd y(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        y(i) = class_dist(gen);
    }
    
    return make_pair(X, y);
}

/**
 * @brief Génère des données linéairement séparables pour un test plus réaliste
 */
pair<MatrixXd, VectorXd> generate_separable_data(int n_samples, int n_features, int n_classes) {
    MatrixXd X(n_samples, n_features);
    VectorXd y(n_samples);
    
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0);
    
    // Créer des clusters séparés pour chaque classe
    for (int cls = 0; cls < n_classes; ++cls) {
        int samples_per_class = n_samples / n_classes;
        int start_idx = cls * samples_per_class;
        int end_idx = (cls == n_classes - 1) ? n_samples : start_idx + samples_per_class;
        
        for (int i = start_idx; i < end_idx; ++i) {
            // Décaler chaque classe dans une direction différente
            for (int j = 0; j < n_features; ++j) {
                X(i, j) = dist(gen) + (cls + 1) * 2.0; // Décalage selon la classe
            }
            y(i) = cls;
        }
    }
    
    return make_pair(X, y);
}

/**
 * @brief Calcule la précision du modèle
 */
double calculate_accuracy(const MatrixXd& predictions, const VectorXd& true_labels) {
    int correct = 0;
    int n_samples = predictions.rows();
    
    for (int i = 0; i < n_samples; ++i) {
        // Trouver la classe prédite (indice avec la plus haute probabilité)
        int predicted_class = 0;
        double max_prob = predictions(i, 0);
        for (int j = 1; j < predictions.cols(); ++j) {
            if (predictions(i, j) > max_prob) {
                max_prob = predictions(i, j);
                predicted_class = j;
            }
        }
        
        // Vérifier si la prédiction est correcte
        if (predicted_class == static_cast<int>(true_labels(i))) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / n_samples * 100.0;
}

int main() {
    try {
        cout << "=== DÉMONSTRATION RÉSEAU DENSE SUR DONNÉES FICTIVES ===" << endl;
        
        // Paramètres du dataset fictif
        const int N_SAMPLES = 10;
        const int N_FEATURES = 20;
        const int N_CLASSES = 3;
        
        // Génération des données
        cout << "Génération des données fictives..." << endl;
        auto [X, y] = generate_separable_data(N_SAMPLES, N_FEATURES, N_CLASSES);
        
        cout << "Dimensions des données:" << endl;
        cout << "  - X: " << X.rows() << " x " << X.cols() << endl;
        cout << "  - y: " << y.size() << " échantillons" << endl;
        cout << "  - Classes: 0 à " << (N_CLASSES - 1) << endl;
        
        // Afficher un échantillon des données
        cout << "\nExtrait des données (premières 5 lignes, 5 colonnes):" << endl;
        cout << X.block(0, 0, 5, 5) << endl;
        cout << "Labels correspondants: ";
        for (int i = 0; i < 5; ++i) {
            cout << static_cast<int>(y(i)) << " ";
        }
        cout << "\n" << endl;
        
        // Architecture du réseau dense
        cout << "=== CONFIGURATION DU RÉSEAU DENSE ===" << endl;
        
        DenseLayer dense1(N_FEATURES, 64);      // Couche cachée 1
        Activation_ReLU activation1;
        
        DenseLayer dense2(64, 5);              // Couche cachée 2
        Activation_ReLU activation2;
        
        DenseLayer dense3(5, N_CLASSES);       // Couche de sortie
        Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
        
        Optimizer_SGD optimizer(0.001);          // Taux d'apprentissage de 0.01
        
        cout << "Architecture du réseau:" << endl;
        cout << "  Input: " << N_FEATURES << " caractéristiques" << endl;
        cout << "  Dense1: " << N_FEATURES << " → 64 neurones" << endl;
        cout << "  ReLU1" << endl;
        cout << "  Dense2: 64 → 32 neurones" << endl;
        cout << "  ReLU2" << endl;
        cout << "  Dense3: 32 → " << N_CLASSES << " neurones (sortie)" << endl;
        cout << "  Softmax + CrossEntropy" << endl;
        cout << "  Optimizer: SGD avec learning_rate=0.01" << endl;
        
        // Entraînement
        cout << "\n=== PHASE D'ENTRAÎNEMENT ===" << endl;
        const int EPOCHS = 100;
        
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            // Forward pass
            dense1.forward(X);
            activation1.forward(dense1.output);
            dense2.forward(activation1.output);
            // std::cout << "\nDense 2 output \n" << dense2.output << std::endl;
            
            activation2.forward(dense2.output);
            // std::cout << "\nActivation 2 output \n" << activation2.output << std::endl;
           
            dense3.forward(activation2.output);
            
            // Calcul de la loss
            double loss = loss_activation.forward(dense3.output, y);
            
            // Calcul de la précision toutes les 10 époques
            double accuracy = 0.0;
            if (epoch % 10 == 0) {
                accuracy = calculate_accuracy(loss_activation.output, y);
            }
            
            // Backward pass
            loss_activation.backward(loss_activation.output, y);
            // std::cout << "\n DInputs from the loss and activation 3 \n" << loss_activation.dinputs;
            // std::cout << "\n inputs in dense3\n" << dense3.inputs;
            // std::cout << "\nPre value \n" << dense3.inputs.transpose() * loss_activation.dinputs;
            dense3.backward(loss_activation.dinputs);
            
            activation2.backward(dense3.dinputs);
            dense2.backward(activation2.dinputs);
            // std::cout << "\nDense 2 dweights \n" << dense2.dweights << std::endl;
            activation1.backward(dense2.dinputs);
            dense1.backward(activation1.dinputs);
            
            // Mise à jour des poids
            optimizer.update_params(dense1);
            
            optimizer.update_params(dense2);
            // cout << "\nPre  Updated " <<dense3.weights.row(2) << std::endl;
            // cout << "\ndweights \n" <<dense3.dweights << std::endl;

            optimizer.update_params(dense3);
            // cout <<"\n Updated " <<dense3.weights.row(2) << std::endl;
            
            // Affichage des résultats
            if (epoch % 10 == 0) {
                cout << "Époque " << epoch << " | Loss: " << loss 
                     << " | Accuracy: " << accuracy << "%" << endl;
            }
        }
        
        // Évaluation finale
        cout << "\n=== ÉVALUATION FINALE ===" << endl;
        dense1.forward(X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);
        loss_activation.forward(dense3.output, y);
        
        double final_accuracy = calculate_accuracy(loss_activation.output, y);
        cout << "Résultats finaux:" << endl;
        cout << "  - Loss finale: " << loss_activation.forward(dense3.output, y) << endl;
        cout << "  - Précision finale: " << final_accuracy << "%" << endl;
        
        // Test de prédiction sur quelques exemples
        cout << "\n=== PRÉDICTIONS SUR 5 EXEMPLES ===" << endl;
        for (int i = 0; i < 5; ++i) {
            int true_class = static_cast<int>(y(i));
            int pred_class = 0;
            double max_prob = loss_activation.output(i, 0);
            for (int j = 1; j < N_CLASSES; ++j) {
                if (loss_activation.output(i, j) > max_prob) {
                    max_prob = loss_activation.output(i, j);
                    pred_class = j;
                }
            }
            
            cout << "Exemple " << i << ": Vrai=" << true_class 
                 << " | Prédit=" << pred_class 
                 << " | Probabilité=" << max_prob * 100 << "%"
                 << " → " << (true_class == pred_class ? "✓" : "✗") << endl;
        }
        
    } catch (const exception& e) {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n=== DÉMONSTRATION TERMINÉE AVEC SUCCÈS ===" << endl;
    return 0;
}