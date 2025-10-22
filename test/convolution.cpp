#include <gtest/gtest.h>
#include <vector>
#include <Eigen/Dense>
#include "../convolution.hpp"

using namespace Eigen;
using namespace std;

// Test de la classe ConvLayer
class ConvLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Configuration commune pour les tests
        input_size = 4;
        input_ch = 1;
        filter_num = 2;
        filter_size = 3;
        padding = 1;
        stride = 1;
    }
    
    int input_size;
    int input_ch;
    int filter_num;
    int filter_size;
    int padding;
    int stride;
};

// Test du constructeur de ConvLayer
TEST_F(ConvLayerTest, ConstructorInitializesCorrectly) {
    ConvLayer conv(input_size, input_ch, filter_num, filter_size, padding, stride);
    
    EXPECT_EQ(conv.input_size, input_size);
    EXPECT_EQ(conv.input_ch, input_ch);
    EXPECT_EQ(conv.filter_size, filter_size);
    EXPECT_EQ(conv.output_ch, filter_num);
    EXPECT_EQ(conv.padding, padding);
    EXPECT_EQ(conv.stride, stride);
    
    // Vérification de la taille de sortie calculée
    int expected_output_size = (input_size + 2*padding - filter_size) / stride + 1;
    EXPECT_EQ(conv.output_size, expected_output_size);
}

// Test de l'initialisation des filtres
TEST_F(ConvLayerTest, FilterInitialization) {
    ConvLayer conv(input_size, input_ch, filter_num, filter_size, padding, stride);
    conv.initialize();
    
    // Vérifier que le bon nombre de filtres est créé
    EXPECT_EQ(conv.filters.size(), filter_num);
    for (const auto& filter_set : conv.filters) {
        EXPECT_EQ(filter_set.size(), input_ch);
        for (const auto& filter : filter_set) {
            EXPECT_EQ(filter.rows(), filter_size);
            EXPECT_EQ(filter.cols(), filter_size);
        }
    }
    
    // Vérifier l'initialisation des biais
    EXPECT_EQ(conv.biases.size(), filter_num);
}

// Test de la propagation avant avec des valeurs connues
TEST_F(ConvLayerTest, ForwardPassWithKnownValues) {
    ConvLayer conv(3, 1, 1, 2, 0, 1); // Taille réduite pour test manuel
    conv.initialize();
    
    // Remplacer le filtre par des valeurs connues
    conv.filters[0][0] = MatrixXd::Constant(2, 2, 1.0); // Filtre tout à 1
    conv.biases[0] = 0.0; // Pas de biais
    
    // Créer une entrée simple
    vector<vector<MatrixXd>> batch_input;
    vector<MatrixXd> input_maps;
    input_maps.push_back(MatrixXd::Constant(3, 3, 2.0)); // Toute l'image à 2
    batch_input.push_back(input_maps);
    
    // Exécuter la propagation avant
    conv.forward(batch_input);
    
    // Vérifier les résultats
    EXPECT_EQ(conv.output_maps.size(), 1); // Batch size = 1
    EXPECT_EQ(conv.output_maps[0].size(), 1); // 1 canal de sortie
    
    // Pour une entrée 3x3 avec filtre 2x2 et stride 1, sortie attendue 2x2
    MatrixXd expected_output = MatrixXd::Constant(2, 2, 8.0); // 2*1 + 2*1 + 2*1 + 2*1 = 8
    EXPECT_TRUE(conv.output_maps[0][0].isApprox(expected_output));
}

// Test avec padding
TEST_F(ConvLayerTest, ForwardPassWithPadding) {
    ConvLayer conv(2, 1, 1, 3, 1, 1); // Entrée 2x2, filtre 3x3, padding 1
    conv.initialize();
    
    // Filtre identité simplifié
    conv.filters[0][0] = MatrixXd::Identity(3, 3);
    conv.biases[0] = 0.0;
    
    vector<vector<MatrixXd>> batch_input;
    vector<MatrixXd> input_maps;
    input_maps.push_back(MatrixXd::Identity(2, 2));
    batch_input.push_back(input_maps);
    
    conv.forward(batch_input);
    
    // Avec padding, la sortie devrait avoir la même taille que l'entrée
    EXPECT_EQ(conv.output_maps[0][0].rows(), 2);
    EXPECT_EQ(conv.output_maps[0][0].cols(), 2);
}

// Test de la classe PoolLayer
class PoolLayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_size = 4;
        input_ch = 2;
        pool_size = 2;
    }
    
    int input_size;
    int input_ch;
    int pool_size;
};

// Test du constructeur de PoolLayer
TEST_F(PoolLayerTest, ConstructorInitializesCorrectly) {
    PoolLayer pool(input_size, input_ch, pool_size);
    
    EXPECT_EQ(pool.input_size, input_size);
    EXPECT_EQ(pool.input_ch, input_ch);
    EXPECT_EQ(pool.pool_size, pool_size);
    
    // Vérification de la taille de sortie calculée
    int expected_output_size = input_size / pool_size;
    EXPECT_EQ(pool.output_size, expected_output_size);
}

// Test du pooling max avec des valeurs connues
TEST_F(PoolLayerTest, MaxPoolingForwardPass) {
    PoolLayer pool(4, 1, 2); // Entrée 4x4, 1 canal, pooling 2x2
    
    // Créer une entrée avec des valeurs connues
    vector<vector<MatrixXd>> batch_input;
    vector<MatrixXd> input_maps;
    
    MatrixXd input(4, 4);
    input << 1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12,
             13, 14, 15, 16;
    input_maps.push_back(input);
    batch_input.push_back(input_maps);
    
    // Exécuter le pooling
    pool.forward(batch_input);
    
    // Vérifier les résultats
    EXPECT_EQ(pool.output_maps.size(), 1); // Batch size = 1
    EXPECT_EQ(pool.output_maps[0].size(), 1); // 1 canal
    
    // Résultat attendu du pooling max 2x2
    MatrixXd expected_output(2, 2);
    expected_output << 6, 8,
                       14, 16;
    
    EXPECT_TRUE(pool.output_maps[0][0].isApprox(expected_output));
}

// Test du flatten
TEST_F(PoolLayerTest, FlattenOutput) {
    PoolLayer pool(2, 2, 2); // Entrée 2x2, 2 canaux, pooling 2x2
    
    vector<vector<MatrixXd>> batch_input;
    vector<MatrixXd> input_maps;
    
    // Deux canaux d'entrée
    input_maps.push_back(MatrixXd::Constant(2, 2, 1.0));
    input_maps.push_back(MatrixXd::Constant(2, 2, 2.0));
    batch_input.push_back(input_maps);
    
    pool.forward(batch_input);
    MatrixXd& flattened = pool.flatten();
    
    // Après pooling 2x2 sur entrée 2x2, chaque canal donne 1x1
    // Avec 2 canaux, le vecteur aplati devrait avoir taille 2
    EXPECT_EQ(flattened.rows(), 2);
    EXPECT_EQ(flattened.cols(), 1);
    
    // Vérifier les valeurs
    EXPECT_DOUBLE_EQ(flattened(0, 0), 1.0); // Max du premier canal
    EXPECT_DOUBLE_EQ(flattened(1, 0), 2.0); // Max du deuxième canal
}

// Test avec batch de plusieurs échantillons
TEST_F(PoolLayerTest, BatchProcessing) {
    PoolLayer pool(2, 1, 2);
    
    vector<vector<MatrixXd>> batch_input;
    
    // Premier échantillon
    vector<MatrixXd> sample1;
    sample1.push_back(MatrixXd::Constant(2, 2, 1.0));
    batch_input.push_back(sample1);
    
    // Deuxième échantillon
    vector<MatrixXd> sample2;
    sample2.push_back(MatrixXd::Constant(2, 2, 2.0));
    batch_input.push_back(sample2);
    
    pool.forward(batch_input);
    
    EXPECT_EQ(pool.output_maps.size(), 2); // Deux échantillons dans le batch
    EXPECT_DOUBLE_EQ(pool.output_maps[0][0](0, 0), 1.0); // Premier échantillon
    EXPECT_DOUBLE_EQ(pool.output_maps[1][0](0, 0), 2.0); // Deuxième échantillon
}

// Test des dimensions avec différentes configurations
TEST_F(ConvLayerTest, VariousConfigurations) {
    // Test 1: Pas de padding
    ConvLayer conv1(5, 3, 4, 3, 0, 1);
    EXPECT_EQ(conv1.output_size, 3); // (5 + 0 - 3)/1 + 1 = 3
    
    // Test 2: Avec padding
    ConvLayer conv2(5, 3, 4, 3, 1, 1);
    EXPECT_EQ(conv2.output_size, 5); // (5 + 2 - 3)/1 + 1 = 5
    
    // Test 3: Stride 2
    ConvLayer conv3(5, 3, 4, 3, 0, 2);
    EXPECT_EQ(conv3.output_size, 2); // (5 + 0 - 3)/2 + 1 = 2
}

// Point d'entrée pour les tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}