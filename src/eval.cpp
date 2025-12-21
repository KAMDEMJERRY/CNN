#include "eval.hpp"

#include <omp.h>


int main(int argc, char* argv[]){    

    int num_thread = 1;
    if(argc > 1){
        std::istringstream iss(argv[1]);
        iss >> num_thread;
        cout << "Nbr threads :: ("<< num_thread << ")\n";
    }
    Eigen::setNbThreads(num_thread);

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
    params.epochs = 50;
    params.learning_rate = 0.001;
    params.decay = 1e-4;
    params.momentum = 0.9;
    params.checkpoint = 5;  // Corrigé: checkpoints -> checkpoint

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
    params.conv1_filter_number = 4;
    params.conv1_filter_size = 5;
    params.conv1_padding = 1;
    params.conv1_stride = 1;
    params.pool1_size = 2;

    // Configuration Conv2 (corrigé: input_channel_number devrait être 8, pas input_channels)
    params.conv2_filter_number = 3; //16;            // 16 filtres comme défini dans conv2
    params.conv2_filter_size = 5;
    params.conv2_padding = 1;
    params.conv2_stride = 1;

    params.pool2_size = 2;

    // Configuration Conv3 (si vous l'utilisez plus tard, sinon vous pouvez supprimer)

    params.conv3_filter_number = 5; //32;
    params.conv3_filter_size = 3;
    params.conv3_padding = 1;
    params.conv3_stride = 1;

    params.pool3_size = 2;

    // Configuration des couches denses
    params.dense2_inputsize = 20; //64;              // Sortie de dense1 = entrée de dense2
    params.dense3_inputsize = 10;              // Sortie de dense2 = entrée de dense3
    params.dense4_inputsize = imgDataset.classes.size(); // Sortie de dense3 = nombre de classes

    CNNModel cnn_model(params);
    cnn_model.compile();






    auto begin = std::chrono::high_resolution_clock::now();

    cnn_model.conv1.forward(inputs_train);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    
    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
    
    return 0;
}


