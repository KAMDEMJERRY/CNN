#include <gtest/gtest.h>
#include <vector>
#include <Eigen/Dense>
#include "imgdataset.hpp"

using namespace Eigen;
using namespace std;

class ImageDatasetTest : public ::testing::Test {
protected:
    ImageDataset dataset;
    
    void SetUp() override {
        // Initialisation des variables de test
        string dataset_path = "../../../dataset/bloodcellsub/images/TRAIN/";
        dataset = ImageDataset(dataset_path);
    }
};

TEST_F(ImageDatasetTest, LoadDataset) {
    // Vérifier que les images et labels sont chargés
    vector<vector<MatrixXd>> images = dataset.getX();
    auto y_data = dataset.getY();
    vector<string> labels = y_data.first;
    
    EXPECT_FALSE(images.empty());
    EXPECT_FALSE(labels.empty());
    EXPECT_EQ(images.size(), labels.size());
}

TEST_F(ImageDatasetTest, OrdinalEncoding) {
    // Vérifier l'encodage ordinal des labels
    auto y_data = dataset.getY();
    vector<string> labels = y_data.first;
    VectorXi encoded_labels = y_data.second;
    
    EXPECT_EQ(labels.size(), encoded_labels.size());
    for (int i = 0; i < encoded_labels.size(); ++i) {
        EXPECT_GE(encoded_labels(i), 0);
        EXPECT_LT(encoded_labels(i), dataset.classes.size());
    }
}

TEST_F(ImageDatasetTest, ShuffleDataset) {
    // Copier les images et labels avant le mélange
    vector<vector<MatrixXd>> original_images = dataset.getX();
    auto original_y_data = dataset.getY();
    vector<string> original_labels = original_y_data.first;
    VectorXi original_encoded_labels = original_y_data.second;
    
    dataset.shuffle_dataset();
    
    vector<vector<MatrixXd>> shuffled_images = dataset.getX();
    auto shuffled_y_data = dataset.getY();
    vector<string> shuffled_labels = shuffled_y_data.first;
    VectorXi shuffled_encoded_labels = shuffled_y_data.second;
    
    // Vérifier que les tailles sont identiques
    EXPECT_EQ(original_images.size(), shuffled_images.size());
    EXPECT_EQ(original_labels.size(), shuffled_labels.size());
    
    // Vérifier que la correspondance entre images et labels est maintenue
    bool is_shuffled = false;
    for (size_t i = 0; i < shuffled_images.size(); ++i) {
        // Trouver l'index original de l'image actuelle
        auto it = find(original_images.begin(), original_images.end(), shuffled_images[i]);
        if (it != original_images.end()) {
            size_t original_index = distance(original_images.begin(), it);
            // Vérifier que le label correspond
            EXPECT_EQ(original_labels[original_index], shuffled_labels[i]);
            EXPECT_EQ(original_encoded_labels(original_index), shuffled_encoded_labels(i));
            
            // Si on trouve au moins une différence de position, c'est mélangé
            if (original_index != i) {
                is_shuffled = true;
            }
        }
    }
    
    // Vérifier que le mélange a effectivement eu lieu
    EXPECT_TRUE(is_shuffled);
}

TEST_F(ImageDatasetTest, TrainTestSplit) {
    // Obtenir les ensembles d'entraînement et de test
    auto train_data = dataset.getTrain();
    auto test_data = dataset.getTest();
    
    vector<vector<MatrixXd>> train_images = train_data.first;
    VectorXi train_labels = train_data.second;
    
    vector<vector<MatrixXd>> test_images = test_data.first;
    VectorXi test_labels = test_data.second;
    
    // Vérifier que les ensembles ne sont pas vides
    EXPECT_FALSE(train_images.empty());
    EXPECT_FALSE(test_images.empty());
    
    // Vérifier que les tailles correspondent au split
    size_t total_size = train_images.size() + test_images.size();
    float expected_train_ratio = static_cast<float>(train_images.size()) / total_size;
    EXPECT_NEAR(expected_train_ratio, dataset.split, 0.1); // Tolérance de 10%
    
    // Vérifier que les ensembles sont disjoints
    for (const auto& train_img : train_images) {
        bool found_in_test = false;
        for (const auto& test_img : test_images) {
            if (train_img == test_img) {
                found_in_test = true;
                break;
            }
        }
        EXPECT_FALSE(found_in_test);
    }
}

TEST_F(ImageDatasetTest, ImageDimensions) {
    // Vérifier que les dimensions des images sont correctes
    vector<vector<MatrixXd>> images = dataset.getX();
    for (const auto& image_channels : images) {
        // Vérifier qu'il y a 3 canaux (RGB)
        EXPECT_EQ(image_channels.size(), 3);
        
        for (const auto& channel : image_channels) {
            // Vérifier les dimensions de chaque canal
            EXPECT_EQ(channel.rows(), dataset.image_size);
            EXPECT_EQ(channel.cols(), dataset.image_size);
        }
    }
}

TEST_F(ImageDatasetTest, Summary) {
    // Vérifier que la méthode summary s'exécute sans erreur
    EXPECT_NO_THROW(dataset.summary());
}

TEST_F(ImageDatasetTest, ClassesConsistency) {
    // Vérifier la cohérence des classes
    auto y_data = dataset.getY();
    vector<string> labels = y_data.first;
    
    // Tous les labels doivent appartenir aux classes définies
    for (const auto& label : labels) {
        auto it = find(dataset.classes.begin(), dataset.classes.end(), label);
        EXPECT_NE(it, dataset.classes.end());
    }
}

TEST_F(ImageDatasetTest, EncodedLabelsConsistency) {
    // Vérifier la cohérence entre les labels encodés et les classes
    auto y_data = dataset.getY();
    vector<string> labels = y_data.first;
    VectorXi encoded_labels = y_data.second;
    
    for (size_t i = 0; i < labels.size(); ++i) {
        // Trouver l'index de la classe dans le vecteur classes
        auto it = find(dataset.classes.begin(), dataset.classes.end(), labels[i]);
        size_t class_index = distance(dataset.classes.begin(), it);
        
        // Vérifier que l'encodage correspond
        EXPECT_EQ(encoded_labels(i), static_cast<int>(class_index));
    }
}

TEST_F(ImageDatasetTest, EmptyDataset) {
    // Tester avec un dataset vide
    ImageDataset empty_dataset;
    
    // Vérifier que les getters retournent des conteneurs vides
    vector<vector<MatrixXd>> empty_images = empty_dataset.getX();
    auto empty_y = empty_dataset.getY();
    
    EXPECT_TRUE(empty_images.empty());
    EXPECT_TRUE(empty_y.first.empty());
    EXPECT_EQ(empty_y.second.size(), 0);
}

TEST_F(ImageDatasetTest, SplitBoundaries) {
    // Tester les limites du split
    auto train_data = dataset.getTrain();
    auto test_data = dataset.getTest();
    
    vector<vector<MatrixXd>> train_images = train_data.first;
    vector<vector<MatrixXd>> test_images = test_data.first;
    
    // Vérifier qu'au moins une image est dans chaque ensemble
    EXPECT_GT(train_images.size(), 0);
    EXPECT_GT(test_images.size(), 0);
    
    // Vérifier que la somme correspond au total
    auto all_images = dataset.getX();
    EXPECT_EQ(train_images.size() + test_images.size(), all_images.size());
}

TEST_F(ImageDatasetTest, ImageDataRange) {
    // Vérifier que les valeurs des pixels sont dans une plage raisonnable
    vector<vector<MatrixXd>> images = dataset.getX();
    
    for (const auto& image_channels : images) {
        for (const auto& channel : image_channels) {
            double min_val = channel.minCoeff();
            double max_val = channel.maxCoeff();
            
            // Les valeurs normalisées devraient être entre 0 et 1
            EXPECT_GE(min_val, 0.0);
            EXPECT_LE(max_val, 1.0);
        }
    }
}

TEST_F(ImageDatasetTest, LoaderIntegration) {
    // Tester l'intégration avec ImageDatasetLoader
    EXPECT_EQ(dataset.images.size(), dataset.loader.getImages().size());
    EXPECT_EQ(dataset.labels.size(), dataset.loader.getLabels().size());
    
    // Vérifier que les dimensions correspondent
    if (!dataset.images.empty()) {
        EXPECT_EQ(dataset.images[0].size(), 3); // 3 canaux RGB
        EXPECT_EQ(dataset.images[0][0].rows(), dataset.image_size);
        EXPECT_EQ(dataset.images[0][0].cols(), dataset.image_size);
    }
}