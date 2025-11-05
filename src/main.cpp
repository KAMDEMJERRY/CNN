#include "convolution.hpp"
#include "dense.hpp"
#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <utility>
#include "model.hpp"
#include <pqxx/pqxx>


using namespace std;
using namespace Eigen;


// #define BASE_DATA_PATH "../../../dataset/bloodcell/images/TRAIN/"
// #define  BASE_DATA_PATH  "../../../dataset/bloodcellsub/images/TRAIN/"
#define  BASE_DATA_PATH  "../../../dataset/bloodcellsub1/images/TRAIN/"

int main(int argc, char*argv[]) {
    try {

        cout << "=== CHARGEMENT DU DATASET ===" <<endl;
        ImageDataset imgDataset(BASE_DATA_PATH, 220, "GRAY");   // RGB or GRAY
        imgDataset.split = 0.8; // 80% pour l'entraînement, 20% pour le test
        std::vector<std::vector<MatrixXd>> inputs_train  = imgDataset.getTrain().first;
        std::vector<std::vector<MatrixXd>> inputs_test  = imgDataset.getTest().first;
        VectorXd y_train = (imgDataset.getTrain().second).cast<double>();
        VectorXd y_test = (imgDataset.getTest()).second.cast<double>();
        int image_size = imgDataset.image_size; // Les images sont carrées (128x128)
        int input_channels = imgDataset.channels; // Images en niveaux de RGB(3) ou de gris (1)


        // Vérifier que des images ont été chargées
        assert(!imgDataset.images.empty() && "Le dataset d'images ne doit pas être vide");
        assert(!imgDataset.labels.empty() && "Les labels ne doivent pas être vides");
        
        // Afficher les informations du dataset
        imgDataset.summary();









        // Définir les paramètres du modèle CNN

        CNNParameters params;
        params.epochs = 100;
        params.learning_rate = 0.001;
        params.decay = 1e-4;
        params.momentum = 0.9;
        params.checkpoint = 1;  // Corrigé: checkpoints -> checkpoint

        // Configuration Conv1
        params.conv1_inputsize = image_size;
        params.conv1_input_channel_number = input_channels;
        params.conv1_filter_number = 8;
        params.conv1_filter_size = 5;
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

        params.conv3_filter_number = 32;
        params.conv3_filter_size = 3;
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
            bool quit = false;
            switch (choice)
            {
                case 0:
                    cnn_model.fit(inputs_train, y_train);
                    cnn_model.dump();
                    break;
                case 1:
                    cnn_model.evaluate(inputs_test, y_test, imgDataset.classes);
                    break;

                case 2:
                    quit = true;
                    break;

                default:
                    break;
            }

            if(quit){
                break;
            }
        }
        
  
    } catch (const std::exception& e) {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}