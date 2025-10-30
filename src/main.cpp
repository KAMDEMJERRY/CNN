#include "convolution.hpp"
#include "dense.hpp"
#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <utility>
#include "model.hpp"

using namespace std;
using namespace Eigen;



int main(int argc, char*argv[]) {
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

        std::vector<std::vector<MatrixXd>> inputs(n_images);
        for (int img_idx = 0; img_idx < n_images; ++img_idx) {
            inputs[img_idx].push_back(imgDataset.images[img_idx]);
        }    
       








        CNNParameters params;
        params.epochs = 100;
        params.learning_rate = 0.05;
        params.checkpoint = 1;  // Corrigé: checkpoints -> checkpoint

        // Configuration Conv1
        params.conv1_inputsize = image_size;
        params.conv1_input_channel_number = input_channels;
        params.conv1_filter_number = 8;
        params.conv1_filter_size = 3;
        params.conv1_padding = 1;
        params.conv1_stride = 1;
        params.pool1_size = 2;

        // Configuration Conv2 (corrigé: input_channel_number devrait être 8, pas input_channels)
        params.conv2_filter_number = 16;            // 16 filtres comme défini dans conv2
        params.conv2_filter_size = 5;
        params.conv2_padding = 1;
        params.conv2_stride = 1;

        params.pool2_size = 2;

        // Configuration Conv3 (si vous l'utilisez plus tard, sinon vous pouvez supprimer)

        params.conv3_filter_number = 8;
        params.conv3_filter_size = 5;
        params.conv3_padding = 1;
        params.conv3_stride = 1;

        params.pool3_size = 2;

        // Configuration des couches denses
        params.dense2_inputsize = 64;              // Sortie de dense1 = entrée de dense2
        params.dense3_inputsize = 32;              // Sortie de dense2 = entrée de dense3
        params.dense4_inputsize = imgDataset.classes.size(); // Sortie de dense3 = nombre de classes

        CNNModel cnn_model(params);
        cnn_model.compile();
        // model.load()

        while(1){
            std::cout << "0: Train | 1: Test | 2: Quit" << std::endl;
            int choice;
            std::cout << ">>> ";
            std::cin >> choice;
            switch (choice)
            {
                case 0:
                    cnn_model.fit(inputs, y);
                    cnn_model.dump();
                    break;
                case 1:
                    cnn_model.evaluate(inputs, y);
                    return 0;
                case 2:
                    break;
                default:
                    break;
            }
        }
        
  
    } catch (const std::exception& e) {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}