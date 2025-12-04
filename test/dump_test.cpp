#include <gtest/gtest.h>
#include <boost/serialization/serialization.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <Eigen/Dense>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>

// Include vos headers
#include "model_repo.hpp"

namespace fs = std::filesystem;



// ============================================================================
// FIXTURES ET HELPERS
// ============================================================================

class SerializationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Nettoyer les fichiers de test existants
        if (fs::exists("test_model.bin")) fs::remove("test_model.bin");
        if (fs::exists("test_model.txt")) fs::remove("test_model.txt");
        if (fs::exists("test_dense.bin")) fs::remove("test_dense.bin");
        if (fs::exists("test_conv.bin")) fs::remove("test_conv.bin");
        if (fs::exists("test_matrix.bin")) fs::remove("test_matrix.bin");
        if (fs::exists("test_matrix.txt")) fs::remove("test_matrix.txt");
        if (fs::exists("integration_test.bin")) fs::remove("integration_test.bin");
    }
    
    void TearDown() override {
        // Nettoyer après les tests
        if (fs::exists("test_model.bin")) fs::remove("test_model.bin");
        if (fs::exists("test_model.txt")) fs::remove("test_model.txt");
        if (fs::exists("test_dense.bin")) fs::remove("test_dense.bin");
        if (fs::exists("test_conv.bin")) fs::remove("test_conv.bin");
        if (fs::exists("test_matrix.bin")) fs::remove("test_matrix.bin");
        if (fs::exists("test_matrix.txt")) fs::remove("test_matrix.txt");
        if (fs::exists("integration_test.bin")) fs::remove("integration_test.bin");
    }
    
    // Helper pour comparer deux matrices Eigen
    template<typename MatrixType>
    void expectMatrixEqual(const MatrixType& m1, const MatrixType& m2, 
                          double tolerance = 1e-6) {
        EXPECT_EQ(m1.rows(), m2.rows());
        EXPECT_EQ(m1.cols(), m2.cols());
        EXPECT_TRUE(m1.isApprox(m2, tolerance)) 
            << "Matrices differ:\n" 
            << "Original:\n" << m1 << "\nLoaded:\n" << m2;
    }
    
    // Helper pour comparer deux vecteurs
    template<typename VectorType>
    void expectVectorEqual(const VectorType& v1, const VectorType& v2,
                          double tolerance = 1e-6) {
        EXPECT_EQ(v1.size(), v2.size());
        EXPECT_TRUE(v1.isApprox(v2, tolerance))
            << "Vectors differ:\n"
            << "Original: " << v1.transpose() << "\n"
            << "Loaded: " << v2.transpose();
    }
    
    // Helper pour créer un batch de test
    std::vector<std::vector<Eigen::MatrixXd>> createTestBatch(int batch_size = 1) {
        std::vector<std::vector<Eigen::MatrixXd>> batch;
        for (int i = 0; i < batch_size; ++i) {
            std::vector<Eigen::MatrixXd> channels;
            channels.push_back(Eigen::MatrixXd::Random(28, 28));
            batch.push_back(channels);
        }
        return batch;
    }
};

// ============================================================================
// TESTS EIGEN SERIALIZATION
// ============================================================================

TEST_F(SerializationTest, EigenMatrixXdSerialization) {
    // Créer une matrice aléatoire
    Eigen::MatrixXd original = Eigen::MatrixXd::Random(5, 3);
    
    // Test sérialisation binaire
    {
        std::ofstream ofs("test_matrix.bin", std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << original;
    }
    
    Eigen::MatrixXd loaded_bin;
    {
        std::ifstream ifs("test_matrix.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loaded_bin;
    }
    
    expectMatrixEqual(original, loaded_bin);
    
    // Test sérialisation texte (pour matrices Eigen, mieux vaut éviter text)
    // car il peut y avoir des problèmes de format
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << original;
    }
    
    Eigen::MatrixXd loaded_stream;
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> loaded_stream;
    }
    
    expectMatrixEqual(original, loaded_stream);
}

TEST_F(SerializationTest, EigenVectorXdSerialization) {
    Eigen::VectorXd original = Eigen::VectorXd::Random(10);
    
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << original;
    }
    
    Eigen::VectorXd loaded;
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> loaded;
    }
    
    expectVectorEqual(original, loaded);
}

// ============================================================================
// TESTS DENSE LAYER
// ============================================================================

TEST_F(SerializationTest, DenseLayerSerialization) {
    // Créer une couche dense
    DenseLayer original(10, 5);  // 10 entrées, 5 neurones
    original.weights = Eigen::MatrixXd::Random(10, 5);
    original.biases = Eigen::VectorXd::Random(5);
    original.weight_regularizer_l1 = 0.01;
    original.weight_regularizer_l2 = 0.02;
    original.bias_regularizer_l1 = 0.001;
    original.bias_regularizer_l2 = 0.002;
    
    // Initialiser les momentum
    original.weights_momentum = Eigen::MatrixXd::Zero(10, 5);
    original.biases_momentum = Eigen::VectorXd::Zero(5);
    
    // Sérialiser
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << original;
    }
    
    // Désérialiser
    DenseLayer loaded(1, 1);  // Taille temporaire
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> loaded;
    }
    
    // Vérifications
    EXPECT_EQ(original.n_inputs, loaded.n_inputs);
    EXPECT_EQ(original.n_neurons, loaded.n_neurons);
    expectMatrixEqual(original.weights, loaded.weights);
    expectVectorEqual(original.biases, loaded.biases);
    EXPECT_DOUBLE_EQ(original.weight_regularizer_l1, loaded.weight_regularizer_l1);
    EXPECT_DOUBLE_EQ(original.weight_regularizer_l2, loaded.weight_regularizer_l2);
    EXPECT_DOUBLE_EQ(original.bias_regularizer_l1, loaded.bias_regularizer_l1);
    EXPECT_DOUBLE_EQ(original.bias_regularizer_l2, loaded.bias_regularizer_l2);
    expectMatrixEqual(original.weights_momentum, loaded.weights_momentum);
    expectVectorEqual(original.biases_momentum, loaded.biases_momentum);
}

// ============================================================================
// TESTS CNN PARAMETERS
// ============================================================================

TEST_F(SerializationTest, CNNParametersSerialization) {
    CNNParameters original;
    
    // Remplir avec des valeurs
    original.learning_rate = 0.01;
    original.decay = 1e-5;
    original.momentum = 0.9;
    original.iterations = 0;
    original.batch_size = 32;
    original.epochs = 100;
    
    original.d_weight_regularizer_l1 = 0.0;
    original.d_weight_regularizer_l2 = 1e-4;
    original.d_bias_regularizer_l1 = 0.0;
    original.d_bias_regularizer_l2 = 1e-4;
    
    original.c_weight_regularizer_l1 = 0.0;
    original.c_weight_regularizer_l2 = 1e-4;
    original.c_bias_regularizer_l1 = 0.0;
    original.c_bias_regularizer_l2 = 1e-4;
    
    original.checkpoint = 10;
    
    original.conv1_inputsize = 28;
    original.conv1_input_channel_number = 1;
    original.conv1_filter_number = 8;
    original.conv1_filter_size = 3;
    original.conv1_padding = 1;
    original.conv1_stride = 1;
    
    original.pool1_size = 2;
    
    original.conv2_filter_number = 0;
    original.conv2_filter_size = 3;
    original.conv2_padding = 1;
    original.conv2_stride = 1;
    
    original.pool2_size = 2;
    
    original.conv3_filter_number = 0;
    original.conv3_filter_size = 3;
    original.conv3_padding = 1;
    original.conv3_stride = 1;
    
    original.pool3_size = 2;
    
    original.dense1_inputsize = 0;
    original.dense2_inputsize = 64;
    original.dense3_inputsize = 10;
    original.dense4_inputsize = 10;
    
    original.dataset_path = "/path/to/dataset";
    original.databaseURL = "postgresql://localhost:5432/cnn";
    original.bd_username = "admin";
    original.bd_password = "secret";
    
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << original;
    }
    
    CNNParameters loaded;
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> loaded;
    }
    
    // Vérifier toutes les valeurs
    EXPECT_DOUBLE_EQ(original.learning_rate, loaded.learning_rate);
    EXPECT_DOUBLE_EQ(original.decay, loaded.decay);
    EXPECT_DOUBLE_EQ(original.momentum, loaded.momentum);
    EXPECT_EQ(original.iterations, loaded.iterations);
    EXPECT_EQ(original.batch_size, loaded.batch_size);
    EXPECT_EQ(original.epochs, loaded.epochs);
    
    EXPECT_DOUBLE_EQ(original.d_weight_regularizer_l1, loaded.d_weight_regularizer_l1);
    EXPECT_DOUBLE_EQ(original.d_weight_regularizer_l2, loaded.d_weight_regularizer_l2);
    EXPECT_DOUBLE_EQ(original.d_bias_regularizer_l1, loaded.d_bias_regularizer_l1);
    EXPECT_DOUBLE_EQ(original.d_bias_regularizer_l2, loaded.d_bias_regularizer_l2);
    
    EXPECT_DOUBLE_EQ(original.c_weight_regularizer_l1, loaded.c_weight_regularizer_l1);
    EXPECT_DOUBLE_EQ(original.c_weight_regularizer_l2, loaded.c_weight_regularizer_l2);
    EXPECT_DOUBLE_EQ(original.c_bias_regularizer_l1, loaded.c_bias_regularizer_l1);
    EXPECT_DOUBLE_EQ(original.c_bias_regularizer_l2, loaded.c_bias_regularizer_l2);
    
    EXPECT_EQ(original.checkpoint, loaded.checkpoint);
    
    EXPECT_EQ(original.conv1_inputsize, loaded.conv1_inputsize);
    EXPECT_EQ(original.conv1_input_channel_number, loaded.conv1_input_channel_number);
    EXPECT_EQ(original.conv1_filter_number, loaded.conv1_filter_number);
    EXPECT_EQ(original.conv1_filter_size, loaded.conv1_filter_size);
    EXPECT_EQ(original.conv1_padding, loaded.conv1_padding);
    EXPECT_EQ(original.conv1_stride, loaded.conv1_stride);
    
    EXPECT_EQ(original.pool1_size, loaded.pool1_size);
    
    EXPECT_EQ(original.conv2_filter_number, loaded.conv2_filter_number);
    EXPECT_EQ(original.conv2_filter_size, loaded.conv2_filter_size);
    EXPECT_EQ(original.conv2_padding, loaded.conv2_padding);
    EXPECT_EQ(original.conv2_stride, loaded.conv2_stride);
    
    EXPECT_EQ(original.pool2_size, loaded.pool2_size);
    
    EXPECT_EQ(original.conv3_filter_number, loaded.conv3_filter_number);
    EXPECT_EQ(original.conv3_filter_size, loaded.conv3_filter_size);
    EXPECT_EQ(original.conv3_padding, loaded.conv3_padding);
    EXPECT_EQ(original.conv3_stride, loaded.conv3_stride);
    
    EXPECT_EQ(original.pool3_size, loaded.pool3_size);
    
    EXPECT_EQ(original.dense1_inputsize, loaded.dense1_inputsize);
    EXPECT_EQ(original.dense2_inputsize, loaded.dense2_inputsize);
    EXPECT_EQ(original.dense3_inputsize, loaded.dense3_inputsize);
    EXPECT_EQ(original.dense4_inputsize, loaded.dense4_inputsize);
    
    EXPECT_EQ(original.dataset_path, loaded.dataset_path);
    EXPECT_EQ(original.databaseURL, loaded.databaseURL);
    EXPECT_EQ(original.bd_username, loaded.bd_username);
    EXPECT_EQ(original.bd_password, loaded.bd_password);
}

// ============================================================================
// TESTS CNN MODEL COMPLETE
// ============================================================================

TEST_F(SerializationTest, CNNModelCompleteSerialization) {
    // Créer des paramètres
    CNNParameters params;
    params.epochs = 10;
    params.learning_rate = 0.01;
    params.decay = 1e-5;
    params.momentum = 0.9;
    params.batch_size = 32;
    
    params.conv1_inputsize = 28;
    params.conv1_input_channel_number = 1;
    params.conv1_filter_number = 4;
    params.conv1_filter_size = 3;
    params.conv1_padding = 1;
    params.conv1_stride = 1;
    params.pool1_size = 2;
    
    params.conv2_filter_number = 0;  // Désactiver conv2 pour simplifier
    params.conv3_filter_number = 0;  // Désactiver conv3
    
    params.dense2_inputsize = 32;
    params.dense3_inputsize = 10;
    params.dense4_inputsize = 10;
    
    // Créer le modèle
    CNNModel original(params);
    original.compile();
    
    // Donner un ID numérique
    original.id = 123;
    
    // Test 1: Sérialisation binaire complète
    {
        std::ofstream ofs("test_model.bin", std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << original;
    }
    
    CNNModel loaded_bin(params);
    {
        std::ifstream ifs("test_model.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loaded_bin;
    }
    
    // Vérifications de base
    EXPECT_EQ(original.id, loaded_bin.id);
    EXPECT_EQ(original.params.epochs, loaded_bin.params.epochs);
    EXPECT_EQ(original.params.learning_rate, loaded_bin.params.learning_rate);
    
    // Vérifier la couche dense1
    EXPECT_EQ(original.dense1.n_inputs, loaded_bin.dense1.n_inputs);
    EXPECT_EQ(original.dense1.n_neurons, loaded_bin.dense1.n_neurons);
    if (original.dense1.weights.size() > 0 && loaded_bin.dense1.weights.size() > 0) {
        expectMatrixEqual(original.dense1.weights, loaded_bin.dense1.weights);
        expectVectorEqual(original.dense1.biases, loaded_bin.dense1.biases);
    }
    
    // Vérifier l'optimizer
    EXPECT_DOUBLE_EQ(original.optimizer.learning_rate, loaded_bin.optimizer.learning_rate);
    EXPECT_DOUBLE_EQ(original.optimizer.momentum, loaded_bin.optimizer.momentum);
}

// ============================================================================
// TESTS DE ROBUSTESSE
// ============================================================================

TEST_F(SerializationTest, SerializationEmptyMatrix) {
    Eigen::MatrixXd empty_matrix(0, 0);
    
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << empty_matrix;
    }
    
    Eigen::MatrixXd loaded;
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> loaded;
    }
    
    EXPECT_EQ(empty_matrix.rows(), loaded.rows());
    EXPECT_EQ(empty_matrix.cols(), loaded.cols());
}

TEST_F(SerializationTest, SerializationLargeMatrix) {
    // Tester avec une grande matrice
    Eigen::MatrixXd large_matrix = Eigen::MatrixXd::Random(100, 100);
    
    std::stringstream ss;
    {
        boost::archive::binary_oarchive oa(ss);
        oa << large_matrix;
    }
    
    Eigen::MatrixXd loaded;
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> loaded;
    }
    
    expectMatrixEqual(large_matrix, loaded, 1e-10);
}

TEST_F(SerializationTest, MultipleSerializationRoundTrip) {
    // Test aller-retour multiple
    CNNParameters params;
    params.epochs = 5;
    params.learning_rate = 0.01;
    params.conv1_inputsize = 28;
    params.conv1_input_channel_number = 1;
    params.conv1_filter_number = 2;
    params.conv1_filter_size = 3;
    params.pool1_size = 2;
    params.conv2_filter_number = 0;
    params.conv3_filter_number = 0;
    params.dense2_inputsize = 16;
    params.dense3_inputsize = 10;
    params.dense4_inputsize = 10;
    
    CNNModel model1(params);
    model1.compile();
    model1.id = 1;
    
    // Premier round
    std::stringstream ss1;
    {
        boost::archive::binary_oarchive oa(ss1);
        oa << model1;
    }
    
    CNNModel model2(params);
    {
        boost::archive::binary_iarchive ia(ss1);
        ia >> model2;
    }
    model2.id = 2;
    
    // Deuxième round
    std::stringstream ss2;
    {
        boost::archive::binary_oarchive oa(ss2);
        oa << model2;
    }
    
    CNNModel model3(params);
    {
        boost::archive::binary_iarchive ia(ss2);
        ia >> model3;
    }
    
    // Vérifier que l'ID a été préservé à travers les rounds
    EXPECT_EQ(1, model1.id);
    EXPECT_EQ(2, model2.id);
    EXPECT_EQ(2, model3.id);  // model3 vient de model2
}

// ============================================================================
// TESTS D'INTÉGRATION
// ============================================================================

TEST_F(SerializationTest, IntegrationSaveLoadFile) {
    // Créer un modèle simple
    CNNParameters params;
    params.epochs = 5;
    params.learning_rate = 0.01;
    params.conv1_inputsize = 28;
    params.conv1_input_channel_number = 1;
    params.conv1_filter_number = 2;
    params.conv1_filter_size = 3;
    params.conv1_padding = 1;
    params.conv1_stride = 1;
    params.pool1_size = 2;
    params.conv2_filter_number = 0;
    params.conv3_filter_number = 0;
    params.dense2_inputsize = 16;
    params.dense3_inputsize = 10;
    params.dense4_inputsize = 10;
    
    CNNModel model(params);
    model.compile();
    model.id = 999;
    
    // Sauvegarder dans un fichier
    {
        std::ofstream ofs("integration_test.bin", std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << model;
    }
    
    // Vérifier que le fichier existe et a une taille > 0
    EXPECT_TRUE(fs::exists("integration_test.bin"));
    EXPECT_GT(fs::file_size("integration_test.bin"), 0);
    
    // Charger depuis le fichier
    CNNModel loaded_model(params);
    {
        std::ifstream ifs("integration_test.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loaded_model;
    }
    
    // Vérifications
    EXPECT_EQ(model.id, loaded_model.id);
    EXPECT_EQ(model.params.epochs, loaded_model.params.epochs);
    EXPECT_EQ(model.params.learning_rate, loaded_model.params.learning_rate);
}

// // ============================================================================
// // MAIN
// // ============================================================================

// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
    
//     // Optionnel: initialisation supplémentaire
//     std::cout << "=== Tests de sérialisation CNN avec Boost ===\n";
    
//     int result = RUN_ALL_TESTS();
    
//     return result;
// }