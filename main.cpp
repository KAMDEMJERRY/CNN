#include "convolution.hpp"
#include "dense.hpp"

int main() {
    try {
        // Charger le dataset d'images
        cout << "=== CHARGEMENT DU DATASET ===" << endl;
        ImageDataset imgDataset = loadDataSet();
        
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

        // Préparer l'input pour la convolution (convertir en vector<MatrixXd>)
        std::vector<MatrixXd> input_maps;
        input_maps.push_back(first_image);
        
        int image_size = first_image.rows(); // Les images sont carrées (128x128)
        int input_channels = 1; // Images en niveaux de gris

        // Création de l'architecture CNN
        cout << "=== CONFIGURATION DU CNN ===" << endl;
        cout << "Taille d'entrée: " << image_size << "x" << image_size << endl;
        cout << "Canaux d'entrée: " << input_channels << endl;
        
        // Première couche de convolution
        ConvLayer conv1(image_size, input_channels, 8, 3, 1, 1); // 8 filtres de 3x3
        cout << "Conv1 - Taille de sortie: " << conv1.output_size << "x" << conv1.output_size << endl;
        cout << "Conv1 - Canaux de sortie: " << conv1.output_ch << endl;
        
        // Première couche de pooling
        PoolLayer pool1(conv1.output_size, conv1.output_ch, 2); // Pooling 2x2
        cout << "Pool1 - Taille de sortie: " << pool1.output_size << "x" << pool1.output_size << endl;
        
        // Deuxième couche de convolution
        ConvLayer conv2(pool1.output_size, pool1.input_ch, 16, 3, 1, 1); // 16 filtres de 3x3
        cout << "Conv2 - Taille de sortie: " << conv2.output_size << "x" << conv2.output_size << endl;
        cout << "Conv2 - Canaux de sortie: " << conv2.output_ch << endl;
        
        // Deuxième couche de pooling
        PoolLayer pool2(conv2.output_size, conv2.output_ch, 2); // Pooling 2x2
        cout << "Pool2 - Taille de sortie: " << pool2.output_size << "x" << pool2.output_size << endl;


        // === TRAITEMENT DE PLUSIEURS IMAGES ===
        cout << "\n=== TRAITEMENT DE PLUSIEURS IMAGES ===" << endl;
        int num_test_images = min(5, (int)imgDataset.images.size()); // Tester sur 5 images max
        
        for (int img_idx = 0; img_idx < num_test_images; ++img_idx) {
            cout << "\n--- Image " << img_idx + 1 << " ---" << endl;
            cout << "Label: " << imgDataset.labels[img_idx] << endl;
            
            // Préparer l'input
            std::vector<MatrixXd> current_input;
            current_input.push_back(imgDataset.images[img_idx]);
            
            // Forward pass
            conv1.forward(current_input);
            pool1.forward(conv1.output_maps);
            conv2.forward(pool1.output_maps);
            pool2.forward(conv2.output_maps);
            pool2.flatten();
            
            cout << "Nombre de caractéristiques extraites: " << pool2.flats_output.size() << endl;
            cout << "Valeur moyenne des caractéristiques: " << pool2.flats_output.mean() << endl;
        }

    } catch (const std::exception& e) {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n=== PROCESSUS TERMINÉ AVEC SUCCÈS ===" << endl;
    return 0;
}