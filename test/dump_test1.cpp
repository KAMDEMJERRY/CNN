// #include <gtest/gtest.h>
// #include <boost/serialization/serialization.hpp>
// #include <boost/archive/binary_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>
// #include <fstream>
// #include <sstream>
// #include <filesystem>
// #include <Eigen/Dense>
// #include <vector>
// #include <random>

// // Include vos headers
// #include "model_repo.hpp"

// namespace fs = std::filesystem;

// class FunctionalEqualityTest : public ::testing::Test
// {
// protected:
//     void SetUp() override
//     {
//         // Initialiser un générateur de nombres aléatoires
//         std::random_device rd;
//         rng.seed(rd());

//         // Nettoyer les fichiers de test
//         if (fs::exists("functional_test.bin"))
//             fs::remove("functional_test.bin");
//         if (fs::exists("predictions_test.bin"))
//             fs::remove("predictions_test.bin");
//         if (fs::exists("training_test.bin"))
//             fs::remove("training_test.bin");
//         if (fs::exists("state_test.bin"))
//             fs::remove("state_test.bin");
//     }

//     void TearDown() override
//     {
//         // Nettoyer après les tests
//         if (fs::exists("functional_test.bin"))
//             fs::remove("functional_test.bin");
//         if (fs::exists("predictions_test.bin"))
//             fs::remove("predictions_test.bin");
//         if (fs::exists("training_test.bin"))
//             fs::remove("training_test.bin");
//         if (fs::exists("state_test.bin"))
//             fs::remove("state_test.bin");
//     }

//     // Helper pour créer des données d'entrée réalistes
//     std::vector<std::vector<Eigen::MatrixXd>> createRandomInputBatch(
//         int batch_size = 8,
//         int image_size = 28,
//         int channels = 1)
//     {

//         std::uniform_real_distribution<double> dist(-1.0, 1.0);
//         std::vector<std::vector<Eigen::MatrixXd>> batch;

//         for (int b = 0; b < batch_size; ++b)
//         {
//             std::vector<Eigen::MatrixXd> image_channels;
//             for (int c = 0; c < channels; ++c)
//             {
//                 Eigen::MatrixXd channel(image_size, image_size);
//                 for (int i = 0; i < image_size; ++i)
//                 {
//                     for (int j = 0; j < image_size; ++j)
//                     {
//                         channel(i, j) = dist(rng);
//                     }
//                 }
//                 image_channels.push_back(channel);
//             }
//             batch.push_back(image_channels);
//         }
//         return batch;
//     }

//     // Helper pour comparer des prédictions
//     void assertPredictionsEqual(
//         const Eigen::MatrixXd &pred1,
//         const Eigen::MatrixXd &pred2,
//         double tolerance = 1e-6,
//         const std::string &message = "")
//     {

//         ASSERT_EQ(pred1.rows(), pred2.rows())
//             << message << " - Rows mismatch: "
//             << pred1.rows() << " vs " << pred2.rows();

//         ASSERT_EQ(pred1.cols(), pred2.cols())
//             << message << " - Cols mismatch: "
//             << pred1.cols() << " vs " << pred2.cols();

//         double max_diff = (pred1 - pred2).cwiseAbs().maxCoeff();
//         double mean_diff = (pred1 - pred2).cwiseAbs().mean();
//         double norm_diff = (pred1 - pred2).norm();

//         // Log détaillé en cas d'échec
//         if (max_diff > tolerance)
//         {
//             std::cout << "\nPrédiction mismatch détecté:" << std::endl;
//             std::cout << "Max difference: " << max_diff << std::endl;
//             std::cout << "Mean difference: " << mean_diff << std::endl;
//             std::cout << "Norm difference: " << norm_diff << std::endl;

//             // Afficher les premières valeurs pour débogage
//             std::cout << "\nPremières valeurs originales: ";
//             for (int i = 0; i < std::min(5, static_cast<int>(pred1.size())); ++i)
//             {
//                 std::cout << pred1.data()[i] << " ";
//             }
//             std::cout << "\nPremières valeurs chargées:  ";
//             for (int i = 0; i < std::min(5, static_cast<int>(pred2.size())); ++i)
//             {
//                 std::cout << pred2.data()[i] << " ";
//             }
//             std::cout << std::endl;
//         }

//         EXPECT_TRUE(pred1.isApprox(pred2, tolerance))
//             << message << " - Predictions differ (max diff: " << max_diff
//             << ", mean diff: " << mean_diff << ", norm diff: " << norm_diff << ")";
//     }

//     // Helper pour vérifier l'état complet d'un modèle
//     void verifyModelStateEquality(const CNNModel &model1, const CNNModel &model2,
//                                   double tolerance = 1e-6)
//     {
//         // Vérifier les paramètres de base
//         EXPECT_EQ(model1.id, model2.id);
//         EXPECT_DOUBLE_EQ(model1.learning_rate, model2.learning_rate);
//         EXPECT_DOUBLE_EQ(model1.decay, model2.decay);
//         EXPECT_DOUBLE_EQ(model1.momentum, model2.momentum);

//         // Vérifier l'optimiseur
//         EXPECT_DOUBLE_EQ(model1.optimizer.learning_rate, model2.optimizer.learning_rate);
//         EXPECT_DOUBLE_EQ(model1.optimizer.current_learning_rate, model2.optimizer.current_learning_rate);
//         EXPECT_DOUBLE_EQ(model1.optimizer.decay, model2.optimizer.decay);
//         EXPECT_EQ(model1.optimizer.iterations, model2.optimizer.iterations);
//         EXPECT_DOUBLE_EQ(model1.optimizer.momentum, model2.optimizer.momentum);

//         // Vérifier les couches denses
//         verifyDenseLayerEquality(model1.dense1, model2.dense1, tolerance, "dense1");
//         verifyDenseLayerEquality(model1.dense2, model2.dense2, tolerance, "dense2");
//         verifyDenseLayerEquality(model1.dense3, model2.dense3, tolerance, "dense3");

//         // Vérifier les couches de convolution si elles existent
//         // (ajouter verifyConvLayerEquality si nécessaire)
//     }

//     void verifyDenseLayerEquality(const DenseLayer &layer1, const DenseLayer &layer2,
//                                   double tolerance, const std::string &name)
//     {
//         EXPECT_EQ(layer1.n_inputs, layer2.n_inputs) << " for layer " << name;
//         EXPECT_EQ(layer1.n_neurons, layer2.n_neurons) << " for layer " << name;

//         if (layer1.weights.size() > 0 && layer2.weights.size() > 0)
//         {
//             EXPECT_TRUE(layer1.weights.isApprox(layer2.weights, tolerance))
//                 << "Weights differ for layer " << name;
//             EXPECT_TRUE(layer1.biases.isApprox(layer2.biases, tolerance))
//                 << "Biases differ for layer " << name;
//             EXPECT_TRUE(layer1.weights_momentum.isApprox(layer2.weights_momentum, tolerance))
//                 << "Weights momentum differ for layer " << name;
//             EXPECT_TRUE(layer1.biases_momentum.isApprox(layer2.biases_momentum, tolerance))
//                 << "Biases momentum differ for layer " << name;
//         }
//     }

//     std::mt19937 rng;
// };

// TEST_F(FunctionalEqualityTest, TrainingContinuationAfterSerialization)
// {
//     // Test si l'entraînement peut continuer après chargement
//     CNNParameters params;
//     params.epochs = 10;
//     params.learning_rate = 0.01;
//     params.decay = 1e-5;
//     params.momentum = 0.9;
//     params.batch_size = 8;
//     params.checkpoint = 1; // Important pour voir les logs à chaque epoch

//     params.conv1_inputsize = 28;
//     params.conv1_input_channel_number = 1;
//     params.conv1_filter_number = 4;
//     params.conv1_filter_size = 3;
//     params.conv1_padding = 1;
//     params.conv1_stride = 1;
//     params.pool1_size = 2;

//     params.conv2_inputsize = 0; // 0 pour désactiver
//     params.conv2_filter_number = 0;
//     params.conv2_filter_size = 3;
//     params.conv2_padding = 1;
//     params.conv2_stride = 1;
//     params.pool2_size = 2;

//     params.conv3_filter_number = 0; // Désactiver conv3

//     params.dense2_inputsize = 32;
//     params.dense3_inputsize = 10;
//     params.dense4_inputsize = 10;

//     // Créer des données d'entraînement factices
//     int num_samples = 32;
//     int num_classes = 10;
//     std::vector<std::string> classes = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};

//     auto X_train = createRandomInputBatch(num_samples, 28, 1);

//     // Créer des labels factices (indices de classe, pas one-hot)
//     // Exemple: [0, 2, 1, 9, 3, ...]
//     std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
//     Eigen::VectorXd y_train(num_samples);
//     for (int i = 0; i < num_samples; ++i)
//     {
//         y_train(i) = class_dist(rng);
//     }

//     // Test 1: Entraîner un modèle pour 3 epochs
//     CNNModel model1(params);
//     model1.compile();

//     // Modifier le nombre d'epochs pour le premier entraînement
//     model1.epochs = 3;

//     std::cout << "\n=== Entraînement initial (3 epochs) ===" << std::endl;
//     model1.fit(X_train, y_train);
//     model1.evaluate(X_train, y_train, classes);
//     // Sauvegarder l'état après 3 epochs
//     {
//         std::ofstream ofs("training_test.bin", std::ios::binary);
//         boost::archive::binary_oarchive oa(ofs);
//         oa << model1;
//     }

//     // Vérifier la taille du fichier
//     EXPECT_TRUE(fs::exists("training_test.bin"));
//     EXPECT_GT(fs::file_size("training_test.bin"), 0);

//     // Test 2: Charger et continuer l'entraînement
//     CNNModel model2(params);
//     {
//         std::ifstream ifs("training_test.bin", std::ios::binary);
//         boost::archive::binary_iarchive ia(ifs);
//         ia >> model2;
//     }
//     model2.evaluate(X_train, y_train, classes);
//     // Continuer l'entraînement pour 2 epochs supplémentaires
//     model2.epochs = 2; // Juste 2 epochs de plus
//     std::cout << "\n=== Reprise d'entraînement (2 epochs supplémentaires) ===" << std::endl;
//     model2.fit(X_train, y_train);

//     // Test 3: Créer un modèle témoin entraîné directement pendant 5 epochs
//     CNNModel model_reference(params);
//     model_reference.compile();
//     model_reference.epochs = 5; // 5 epochs d'affilée

//     std::cout << "\n=== Entraînement référence (5 epochs continues) ===" << std::endl;
//     model_reference.fit(X_train, y_train);

//     // Vérifier que les prédictions sont similaires
//     auto test_batch = createRandomInputBatch(4, 28, 1);

//     Eigen::MatrixXd pred_model2;
//     try
//     {
//         pred_model2 = model2.predict(test_batch);
//     }
//     catch (const std::exception &e)
//     {
//         std::cout << "Erreur lors de la prédiction avec model2: " << e.what() << std::endl;
//         // Créer une prédiction factice pour éviter le crash
//         pred_model2 = Eigen::MatrixXd::Random(4, num_classes);
//     }

//     Eigen::MatrixXd pred_reference;
//     try
//     {
//         pred_reference = model_reference.predict(test_batch);
//     }
//     catch (const std::exception &e)
//     {
//         std::cout << "Erreur lors de la prédiction avec model_reference: " << e.what() << std::endl;
//         pred_reference = Eigen::MatrixXd::Random(4, num_classes);
//     }

//     // Tolérance plus grande car les modèles peuvent diverger légèrement
//     // lors de la poursuite de l'entraînement
//     assertPredictionsEqual(pred_model2, pred_reference, 1e-5,
//                            "Models trained for 5 epochs (3+2 vs 5 straight) should have similar predictions");

//     // Test 4: Vérifier l'état de l'optimiseur
//     // Après 5 epochs (3 + 2), l'optimiseur devrait avoir fait 5 itérations
//     EXPECT_GE(model2.optimizer.iterations, 5)
//         << "Optimizer iterations should reflect total training epochs";

//     // Test 5: Vérifier que les poids ne sont pas tous à zéro
//     if (model2.dense1.weights.size() > 0)
//     {
//         double mean_weight = model2.dense1.weights.mean();
//         EXPECT_GT(std::abs(mean_weight), 1e-10)
//             << "Model weights should be non-zero after training";
//     }
// }

// TEST_F(FunctionalEqualityTest, PredictionsIdenticalAfterSerialization)
// {
//     // Créer un modèle simple
//     CNNParameters params;
//     params.epochs = 0; // Pas d'entraînement, juste compilation
//     params.learning_rate = 0.01;
//     params.decay = 1e-5;
//     params.momentum = 0.9;
//     params.batch_size = 4;

//     params.conv1_inputsize = 28;
//     params.conv1_input_channel_number = 1;
//     params.conv1_filter_number = 4;
//     params.conv1_filter_size = 3;
//     params.conv1_padding = 1;
//     params.conv1_stride = 1;
//     params.pool1_size = 2;

//     params.conv2_filter_number = 0;
//     params.conv3_filter_number = 0;

//     params.dense2_inputsize = 32;
//     params.dense3_inputsize = 10;
//     params.dense4_inputsize = 10;

//     CNNModel original(params);
//     original.compile();
//     original.id = 1001;

//     // Générer des données d'entrée
//     auto test_batch = createRandomInputBatch(4, 28, 1);

//     // Faire des prédictions avec le modèle original
//     Eigen::MatrixXd original_predictions;
//     try
//     {
//         original_predictions = original.predict(test_batch);
//         std::cout << "Shape des prédictions originales: "
//                   << original_predictions.rows() << "x"
//                   << original_predictions.cols() << std::endl;
//     }
//     catch (const std::exception &e)
//     {
//         std::cout << "Erreur lors de la prédiction originale: " << e.what() << std::endl;
//         ADD_FAILURE() << "Original model prediction failed: " << e.what();
//         return;
//     }

//     // Sauvegarder le modèle
//     {
//         std::ofstream ofs("functional_test.bin", std::ios::binary);
//         boost::archive::binary_oarchive oa(ofs);
//         oa << original;
//     }

//     // Charger le modèle
//     CNNModel loaded(params);
//     {
//         loaded.load("functional_test.bin");
//         // std::ifstream ifs("functional_test.bin", std::ios::binary);
//         // boost::archive::binary_iarchive ia(ifs);
//         // ia >> loaded;
//     }

//     // Faire des prédictions avec le modèle chargé
//     Eigen::MatrixXd loaded_predictions;
//     try
//     {
//         loaded_predictions = loaded.predict(test_batch);
//         std::cout << "Shape des prédictions chargées: "
//                   << loaded_predictions.rows() << "x"
//                   << loaded_predictions.cols() << std::endl;
//     }
//     catch (const std::exception &e)
//     {
//         std::cout << "Erreur lors de la prédiction chargée: " << e.what() << std::endl;
//         ADD_FAILURE() << "Loaded model prediction failed: " << e.what();
//         return;
//     }

//     // Vérifier que les prédictions sont identiques
//     assertPredictionsEqual(original_predictions, loaded_predictions, 1e-9,
//                            "Predictions should be identical after serialization");
// }

