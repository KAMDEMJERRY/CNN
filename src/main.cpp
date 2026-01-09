#include "convolution.hpp"
#include "dense.hpp"
#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <utility>
#include "model.hpp"

using namespace std;
using namespace Eigen;

// #define  BASE_DATA_PATH "../../../dataset/bloodcell/images/TRAIN/"
// #define  BASE_DATA_PATH  "../../../dataset/bloodcellsub/images/TRAIN/"
// #define  BASE_DATA_PATH  "../../../dataset/bloodcellsub1/images/TRAIN/"
// #define  BASE_DATA_PATH  "../../../dataset/mnist_img/trainingSet/trainingSet"
#define BASE_DATA_PATH "../../../dataset/mnist_img/trainingSample/trainingSample"
#define TEST_DATA_PATH "../../../dataset/mnist_img/trainingSet/trainingSet"
int main(int argc, char *argv[])
{
    try
    
    
    {

        cout << "=== CHARGEMENT DU DATASET ===" << endl;
        ImageDataset imgDataset(BASE_DATA_PATH, 28, "GRAY"); // RGB or GRAY
        ImageDataset imgDatasetEval(TEST_DATA_PATH, 28, "GRAY", 50); // RGB or GRAY
        
        imgDataset.split = 1; 
        imgDatasetEval.split = 0; 

        std::vector<std::vector<MatrixXd>> inputs_train = imgDataset.getTrain().first;
        std::vector<std::vector<MatrixXd>> inputs_test = imgDatasetEval.getTest().first;
        VectorXd y_train = (imgDataset.getTrain().second).cast<double>();
        VectorXd y_test = (imgDatasetEval.getTest().second).cast<double>();
        int image_size = imgDataset.image_size; // Les images sont carrées (128x128)
        int input_channels = imgDataset.channels; // Images en niveaux de RGB(3) ou de gris (1)

        // Vérifier que des images ont été chargées
        assert(!imgDataset.images.empty() && "Le dataset d'images ne doit pas être vide");
        assert(!imgDataset.labels.empty() && "Les labels ne doivent pas être vides");

        // Afficher les informations du dataset
        imgDataset.summary();

        // Définir les paramètres du modèle CNN
        CNNParameters params;
        params.epochs = 10;
        params.learning_rate = 0.01;
        params.decay = 1e-5;
        params.momentum = 0.9;
        params.checkpoint = 1; // Corrigé: checkpoints -> checkpoint

        params.d_weight_regularizer_l1 = 0;
        params.d_weight_regularizer_l2 = 1e-4;
        params.d_bias_regularizer_l1 = 0;
        params.d_bias_regularizer_l2 = 1e-4;

        params.c_weight_regularizer_l1 = 0;
        params.c_weight_regularizer_l2 = 1e-4;
        params.c_bias_regularizer_l1 = 0;
        params.c_bias_regularizer_l2 = 1e-4;

        // Configuration Conv1
        params.conv1_inputsize = image_size;
        params.conv1_input_channel_number = input_channels;
        params.conv1_filter_number = 8;
        params.conv1_filter_size = 3;
        params.conv1_padding = 1;
        params.conv1_stride = 1;
        params.pool1_size = 2;

        // Configuration Conv2 (corrigé: input_channel_number devrait être 8, pas input_channels)
        params.conv2_filter_number = 5; // 16 filtres comme défini dans conv2
        params.conv2_filter_size = 5;
        params.conv2_padding = 1;
        params.conv2_stride = 1;
        params.pool2_size = 2;

        // Configuration Conv3 (si vous l'utilisez plus tard, sinon vous pouvez supprimer)
        params.conv3_filter_number = 5; // 32;
        params.conv3_filter_size = 4;
        params.conv3_padding = 1;
        params.conv3_stride = 1;

        params.pool3_size = 2;

        // Configuration des couches denses
        params.dense2_inputsize = 64;                        // 64;              // Sortie de dense1 = entrée de dense2
        params.dense3_inputsize = 10;                        // Sortie de dense2 = entrée de dense3
        params.dense4_inputsize = imgDataset.classes.size(); // Sortie de dense3 = nombre de classes

        CNNModel *cnn_model = nullptr;

        int choice;
        std::cout << "\nLoad the model from previous archive ?? 0: Yes  | 1: No" << std::endl;
        std::cout << ">>> ";
        std::cin >> choice;

        if (choice == 0)
        {
            std::string defaultfilepath = "model.bin";
            std::string filepath = "";
            std::cout << "\nEnter the filename or hit enter to load from default file" << std::endl;
            std::cout << ">>> ";

            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::getline(std::cin, filepath);

            if (filepath.empty())
            {
                filepath = defaultfilepath;
            }

            // SOLUTION: Créer un modèle SANS le compiler d'abord
            cnn_model = new CNNModel(); // Constructeur par défaut
            cnn_model->load(filepath);

            std::cout << "Model loaded from: " << filepath << std::endl;

            {
                cnn_model->epochs = params.epochs;
                // Dans votre main(), APRÈS le chargement
                std::cout << "\n=== DIAGNOSTIC DES DIMENSIONS ===" << std::endl;
                std::cout << "conv1.input_size: " << cnn_model->conv1.input_size << std::endl;
                std::cout << "conv1.input_ch: " << cnn_model->conv1.input_ch << std::endl;
                std::cout << "conv1.output_ch: " << cnn_model->conv1.output_ch << std::endl;
                std::cout << "conv1.filter_size: " << cnn_model->conv1.filter_size << std::endl;
                
                std::cout << "pool1.input_ch: " << cnn_model->pool1.input_ch << std::endl;
                std::cout << "pool1.input_size: " << cnn_model->pool1.input_size << std::endl;
                std::cout << "pool1.pool_size: " << cnn_model->pool1.pool_size << std::endl;
                std::cout << "pool1.output_size: " << cnn_model->pool1.output_size << std::endl;
                
                std::cout << "\nconv2.input_size: " << cnn_model->conv2.input_size << std::endl;
                std::cout << "conv2.input_ch: " << cnn_model->conv2.input_ch << std::endl;
                std::cout << "conv2.output_ch: " << cnn_model->conv2.output_ch << std::endl;

                std::cout << "\nconv3.input_size: " << cnn_model->conv3.input_size << std::endl;
                std::cout << "conv3.input_ch: " << cnn_model->conv3.input_ch << std::endl;
                std::cout << "conv3.output_ch: " << cnn_model->conv3.output_ch << std::endl;

                // Vérifier les filtres
                std::cout << "\nconv1.filters size: " << cnn_model->conv1.filters.size() << std::endl;
                if (!cnn_model->conv1.filters.empty())
                {
                    std::cout << "conv1.filters[0] shape: ("
                              << cnn_model->conv1.filters[0].size() << "," 
                              << cnn_model->conv1.filters[0][0].rows() << ", "
                              << cnn_model->conv1.filters[0][0].cols() << ")" << std::endl;
                }

                // Vérifier les données d'entrée de test
                if (!inputs_test.empty() && !inputs_test[0].empty())
                {
                    std::cout << "\nTest input shape: ("
                              << inputs_test[0][0].rows() << ", "
                              << inputs_test[0][0].cols() << ")" << std::endl;
                    std::cout << "Test input channels: " << inputs_test[0].size() << std::endl;
                }
            }
        }
        else
        {
            // Créer un nouveau modèle
            cnn_model = new CNNModel(params);
            cnn_model->compile();
        }

        while (1)
        {
            std::cout << "\n0: Train | 1: Test | 2: Quit" << std::endl;
            int choice;
            std::cout << ">>> ";
            std::cin >> choice;
            bool quit = false;
            switch (choice)
            {
            case 0:
            {
                TIMER_START(start_train);
                cnn_model->fit(inputs_train, y_train);
                TIMER_END(start_train);
                TIMER_PRINT(start_train, "Training time");

                {
                    int choice;
                    std::cout << "\nDump the model ?? 0: Yes  | 1: No" << std::endl;
                    std::cout << ">>> ";
                    std::cin >> choice;

                    if (choice == 0)
                    {
                        std::string defaultfilepath = "model.bin";
                        std::string filepath = "";

                        std::cout << "\n Enter the filename or hit enter to save to default file (" << filepath << ")" << std::endl;
                        std::cout << ">>> ";

                        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                        std::getline(std::cin, filepath);

                        if (filepath.empty())
                        {
                            filepath = defaultfilepath;
                        }

                        std::cout << "Saving model to : " << filepath << std::endl;
                        cnn_model->dump(filepath);
                        std::cout << "Model stored into : " << filepath << std::endl;
                    }
                }

                break;
            }

            case 1:
            {
                TIMER_START(start_eval);
                cnn_model->evaluate(inputs_test, y_test, imgDataset.classes);
                TIMER_END(start_eval);
                TIMER_PRINT(start_eval, "Evaluation time");
                break;
            }

            case 2:
                quit = true;
                break;

            default:
                break;
            }

            if (quit)
            {
                delete (cnn_model);
                break;
            }
        }
    }

    catch (const std::exception &e)
    {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    return 0;
}