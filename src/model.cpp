#include "model.hpp"
#include "model_repo.hpp"
#include "utils.hpp"

// #include <filesystem>
// #include <iostream>

namespace fs = std::filesystem;

// Cette fonction sauvegarde dans un fichier une images des filtres de convolutions
void visualizeFilters(CNNModel &model, bool use_grayscale)
{
    cout << "========================================" << endl;
    cout << "VISUALISATION DES FILTRES CNN" << endl;
    cout << "========================================" << endl;

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string timeStamp =  std::ctime(&now_time) + string("  ");
    // Visualisation en niveau de gris
    showFilterEnhanced(string(timeStamp), " convlayer 1", model.conv1.filters, use_grayscale);
    showFilterEnhanced(string(timeStamp), " convlayer 2", model.conv2.filters, use_grayscale);
    showFilterEnhanced(string(timeStamp), " convlayer 3", model.conv3.filters, use_grayscale);

    // Vous pouvez aussi visualiser en couleur ET en niveau de gris
    if (use_grayscale)
    {
        // Afficher aussi en couleur pour comparaison
        showFilterEnhanced(string(timeStamp), " convlayer 1_color", model.conv1.filters, false);
        showFilterEnhanced(string(timeStamp), " convlayer 2_color", model.conv2.filters, false);
        showFilterEnhanced(string(timeStamp), " convlayer 3_color", model.conv3.filters, false);
    }

    cout << "\nVisualisation terminée. Images sauvegardées." << endl;
    cv::waitKey(0);
}

double calculate_accuracy(const MatrixXd &predictions, const VectorXd &true_labels)
{
    int correct = 0;
    int n_samples = predictions.rows();

    for (int i = 0; i < n_samples; ++i)
    {
        // Trouver la classe prédite (indice avec la plus haute probabilité)
        int predicted_class = 0;
        double max_prob = predictions(i, 0);
        for (int j = 1; j < predictions.cols(); ++j)
        {
            if (predictions(i, j) > max_prob)
            {
                max_prob = predictions(i, j);
                predicted_class = j;
            }
        }

        // Vérifier si la prédiction est correcte
        if (predicted_class == static_cast<int>(true_labels(i)))
        {
            correct++;
        }
    }

    return static_cast<double>(correct) / n_samples * 100.0;
}

CNNModel::CNNModel(CNNParameters &params)
    : params(params),
      conv1(), conv2(), conv3(), // You'll need to add default constructors to your layer classes
      pool1(), pool2(), pool3(),
      dense1(), dense2(), dense3(),
      conv1_activation(), conv2_activation(), conv3_activation(),
      activation1(), activation2(),
      loss_activation(),
      optimizer(params.learning_rate, params.decay, params.momentum)

{
    decay = params.decay;
    momentum = params.momentum;
    double d_weight_regularizer_l1 = params.d_weight_regularizer_l1;
    double d_weight_regularizer_l2 = params.d_weight_regularizer_l2;
    double d_bias_regularizer_l1 = params.d_weight_regularizer_l1;
    double d_bias_regularizer_l2 = params.d_weight_regularizer_l2;

    double c_weight_regularizer_l1 = params.c_weight_regularizer_l1;
    double c_weight_regularizer_l2 = params.c_weight_regularizer_l2;
    double c_bias_regularizer_l1 = params.c_weight_regularizer_l1;
    double c_bias_regularizer_l2 = params.c_weight_regularizer_l2;
}

void CNNModel::sethyperparams(CNNParameters &params)
{
    this->params = params;
    // Convolution Layers
    conv1.weight_regularizer_l1 = params.c_weight_regularizer_l1;
    conv1.weight_regularizer_l2 = params.c_weight_regularizer_l2;
    conv1.bias_regularizer_l1 = params.c_weight_regularizer_l1;
    conv1.bias_regularizer_l2 = params.c_weight_regularizer_l2;

    conv2.weight_regularizer_l1 = params.c_weight_regularizer_l1;
    conv2.weight_regularizer_l2 = params.c_weight_regularizer_l2;
    conv2.bias_regularizer_l1 = params.c_weight_regularizer_l1;
    conv2.bias_regularizer_l2 = params.c_weight_regularizer_l2;

    conv3.weight_regularizer_l1 = params.c_weight_regularizer_l1;
    conv3.weight_regularizer_l2 = params.c_weight_regularizer_l2;
    conv3.bias_regularizer_l1 = params.c_weight_regularizer_l1;
    conv3.bias_regularizer_l2 = params.c_weight_regularizer_l2;

    // Dense Layers
    dense1.weight_regularizer_l1 = params.d_weight_regularizer_l1;
    dense1.weight_regularizer_l2 = params.d_weight_regularizer_l2;
    dense1.bias_regularizer_l1 = params.d_weight_regularizer_l1;
    dense1.bias_regularizer_l2 = params.d_weight_regularizer_l2;

    dense2.weight_regularizer_l1 = params.d_weight_regularizer_l1;
    dense2.weight_regularizer_l2 = params.d_weight_regularizer_l2;
    dense2.bias_regularizer_l1 = params.d_weight_regularizer_l1;
    dense2.bias_regularizer_l2 = params.d_weight_regularizer_l2;

    dense3.weight_regularizer_l1 = params.d_weight_regularizer_l1;
    dense3.weight_regularizer_l2 = params.d_weight_regularizer_l2;
    dense3.bias_regularizer_l1 = params.d_weight_regularizer_l1;
    dense3.bias_regularizer_l2 = params.d_weight_regularizer_l2;

    learning_rate = params.learning_rate;
    momentum = params.momentum;
    decay = params.decay;
    epochs = params.epochs;
    checkpoint = params.checkpoint;
}

void CNNModel::compile()
{
    conv1 = ConvLayer(params.conv1_inputsize,            // conv1.inputsize
                      params.conv1_input_channel_number, // conv1_number of channel of an input
                      params.conv1_filter_number,        // conv1.number of filter
                      params.conv1_filter_size,          // conv1_size of a filter
                      params.conv1_stride,               // conv1_stride
                      params.conv1_padding);             // conv1_padding

    pool1 = PoolLayer(conv1.output_size, // pool1_size
                      conv1.output_ch,
                      params.pool1_size);

    conv2 = ConvLayer(pool1.output_size,
                      pool1.input_ch,
                      params.conv2_filter_number,
                      params.conv2_filter_size,
                      params.conv2_stride,
                      params.conv2_padding);

    pool2 = PoolLayer(conv2.output_size,
                      conv2.output_ch, // because output channels of conv2 = input channels of pool2
                      params.pool2_size);

    conv3 = ConvLayer(pool2.output_size,
                      pool2.input_ch, // input_ch == output_ch
                      params.conv3_filter_number,
                      params.conv3_filter_size,
                      params.conv3_stride,
                      params.conv3_padding);

    pool3 = PoolLayer(conv3.output_size,
                      conv3.output_ch, // because output channels of conv3 = input channels of pool3
                      params.pool3_size);

    int input_size = std::pow(pool3.output_size, 2) * pool3.input_ch;
    dense1 = DenseLayer(input_size, params.dense2_inputsize);
    dense2 = DenseLayer(dense1.n_neurons, params.dense3_inputsize);
    dense3 = DenseLayer(dense2.n_neurons, params.dense4_inputsize);

    // Regularization params

    // Convolution Layers
    conv1.weight_regularizer_l1 = params.c_weight_regularizer_l1;
    conv1.weight_regularizer_l2 = params.c_weight_regularizer_l2;
    conv1.bias_regularizer_l1 = params.c_weight_regularizer_l1;
    conv1.bias_regularizer_l2 = params.c_weight_regularizer_l2;

    conv2.weight_regularizer_l1 = params.c_weight_regularizer_l1;
    conv2.weight_regularizer_l2 = params.c_weight_regularizer_l2;
    conv2.bias_regularizer_l1 = params.c_weight_regularizer_l1;
    conv2.bias_regularizer_l2 = params.c_weight_regularizer_l2;

    conv3.weight_regularizer_l1 = params.c_weight_regularizer_l1;
    conv3.weight_regularizer_l2 = params.c_weight_regularizer_l2;
    conv3.bias_regularizer_l1 = params.c_weight_regularizer_l1;
    conv3.bias_regularizer_l2 = params.c_weight_regularizer_l2;

    // Dense Layers
    dense1.weight_regularizer_l1 = params.d_weight_regularizer_l1;
    dense1.weight_regularizer_l2 = params.d_weight_regularizer_l2;
    dense1.bias_regularizer_l1 = params.d_weight_regularizer_l1;
    dense1.bias_regularizer_l2 = params.d_weight_regularizer_l2;

    dense2.weight_regularizer_l1 = params.d_weight_regularizer_l1;
    dense2.weight_regularizer_l2 = params.d_weight_regularizer_l2;
    dense2.bias_regularizer_l1 = params.d_weight_regularizer_l1;
    dense2.bias_regularizer_l2 = params.d_weight_regularizer_l2;

    dense3.weight_regularizer_l1 = params.d_weight_regularizer_l1;
    dense3.weight_regularizer_l2 = params.d_weight_regularizer_l2;
    dense3.bias_regularizer_l1 = params.d_weight_regularizer_l1;
    dense3.bias_regularizer_l2 = params.d_weight_regularizer_l2;

    learning_rate = params.learning_rate;
    momentum = params.momentum;
    decay = params.decay;
    epochs = params.epochs;
    checkpoint = params.checkpoint;
}

void CNNModel::fit(std::vector<std::vector<MatrixXd>> &inputs, VectorXd &y)
{   

    cout << "Taille d'entrée: " << inputs[0][0].rows() << "x" << inputs[0][0].cols() << endl;

    cout << "\n=== PHASE D'ENTRAÎNEMENT ===" << endl;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        params.iterations++;

        // Forward pass
        cout << "\nEpoch :" << epoch + 1 << "/" << epochs << "\n";
        conv1.forward(inputs);
        cout << "Après conv1: " << conv1.output_maps[0][0].rows() << "x" << conv1.output_maps[0][0].cols() << endl;
        conv1_activation.forward(conv1.output_maps);

        pool1.forward(conv1_activation.outputs);
        cout << "Après pool1: " << pool1.output_maps[0][0].rows() << "x" << pool1.output_maps[0][0].cols() << endl;

        conv2.forward(pool1.output_maps);
        cout << "Après conv2: " << conv2.output_maps[0][0].rows() << "x" << conv2.output_maps[0][0].cols() << endl;
        conv2_activation.forward(conv2.output_maps);
        pool2.forward(conv2_activation.outputs);
        cout << "Après pool2: " << pool2.output_maps[0][0].rows() << "x" << pool2.output_maps[0][0].cols() << endl;

        conv3.forward(pool2.output_maps);
        cout << "Après conv3: " << conv3.output_maps[0][0].rows() << "x" << conv3.output_maps[0][0].cols() << endl;

        conv3_activation.forward(conv3.output_maps);
        pool3.forward(conv3_activation.outputs);
        cout << "Après pool3: " << pool3.output_maps[0][0].rows() << "x" << pool3.output_maps[0][0].cols() << endl;

        MatrixXd X;
        X = pool3.flatten();
        cout << "Après Flatten: " << X.rows() << "x" << X.cols() << endl;

        dense1.forward(X);
        cout << "Après dense1: " << dense1.output.rows() << "x" << dense1.output.cols() << endl;

        activation1.forward(dense1.output);

        dense2.forward(activation1.output);
        cout << "Après dense2: " << dense2.output.rows() << "x" << dense2.output.cols() << endl;

        activation2.forward(dense2.output);

        dense3.forward(activation2.output);
        cout << "Après dense3: " << dense3.output.rows() << "x" << dense3.output.cols() << endl;

        // Calcul de la loss
        double data_loss = loss_activation.forward(dense3.output, y);
        double regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2) +
                                     loss_activation.loss.regularization_loss(dense3) + loss_activation.loss.regularization_loss(conv1) +
                                     loss_activation.loss.regularization_loss(conv2) + loss_activation.loss.regularization_loss(conv3);

        double loss = data_loss + regularization_loss;

        // Calcul de l'accuracy toutes les 10 époques
        double accuracy = 0.0;
        if (epoch % params.checkpoint == 0)
        {
            accuracy = calculate_accuracy(loss_activation.output, y);
        }

        // Backward pass
        loss_activation.backward(loss_activation.output, y);
        dense3.backward(loss_activation.dinputs);
        cout << "Apres dense3 backward: " << dense3.dinputs.rows() << "x" << dense3.dinputs.cols() << endl;

        activation2.backward(dense3.dinputs);
        dense2.backward(activation2.dinputs);
        cout << "Apres dense2 backward: " << dense2.dinputs.rows() << "x" << dense2.dinputs.cols() << endl;

        activation1.backward(dense2.dinputs);
        dense1.backward(activation1.dinputs);
        cout << "Apres dense1 backward: " << dense1.dinputs.rows() << "x" << dense1.dinputs.cols() << endl;

        pool3.backward(pool3.unflatten(dense1.dinputs));
        cout << "Apres pool3 backward: " << pool3.dinput[0][0].rows() << "x" << pool3.dinput[0][0].cols() << endl;

        conv3_activation.backward(pool3.dinput);
        // cout << "Apres conv3_activation backward: " << conv3_activation.dinputs[0][0].rows() << "x" << conv3_activation.dinputs[0][0].cols() << endl;

        conv3.backward(conv3_activation.dinputs);
        cout << "Apres conv3 backward: " << conv3.dinputs[0][0].rows() << "x" << conv3.dinputs[0][0].cols() << endl;

        pool2.backward(conv3.dinputs);
        cout << "Apres pool2 backward: " << pool2.dinput[0][0].rows() << "x" << pool2.dinput[0][0].cols() << endl;

        conv2_activation.backward(pool2.dinput);
        // cout << "Apres conv2_activation backward: " << conv2_activation.dinputs[0][0].rows() << "x" << conv2_activation.dinputs[0][0].cols() << endl;

        conv2.backward(conv2_activation.dinputs);
        cout << "Apres conv2 backward: " << conv2.dinputs[0][0].rows() << "x" << conv2.dinputs[0][0].cols() << endl;

        pool1.backward(conv2.dinputs);
        cout << "Apres pool1 backward: " << pool1.dinput[0][0].rows() << "x" << pool1.dinput[0][0].cols() << endl;

        conv1_activation.backward(pool1.dinput);
        // cout << "Apres conv1_activation backward: " << conv1_activation.dinputs[0][0].rows() << "x" << conv1_activation.dinputs[0][0].cols() << endl;

        conv1.backward(pool1.dinput);
        cout << "Apres conv1 backward: " << conv1.dinputs[0][0].rows() << "x" << conv1.dinputs[0][0].cols() << endl;

        // Mise à jour des poids
        optimizer.pre_update_params();
        optimizer.update_params(dense1);
        optimizer.update_params(dense2);
        optimizer.update_params(dense3);
        optimizer.update_params(conv1);
        optimizer.update_params(conv2);
        optimizer.update_params(conv3);
        optimizer.post_update_params();
        // Affichage des résultats
        if (epoch % params.checkpoint == 0)
        {
            cout << "Époque " << epoch
                 << " | Loss: " << loss << "("
                 << "data_loss: " << data_loss << ", "
                 << "reg_loss: " << regularization_loss << ") "
                 << " | Accuracy: " << accuracy << "%"
                 << " | lr: " << optimizer.current_learning_rate << endl;

            dump_metrics(this->params.iterations, loss, accuracy);
        }
    }

    cout << "Visualize " << std::endl;
    visualizeFilters(*this, true);
}

void CNNModel::evaluate(std::vector<std::vector<MatrixXd>> &inputs, VectorXd &Y, vector<string> &classes)
{
    cout << "\n=== PHASE D'ÉVALUATION ===" << endl;

    int correct_predictions = 0;
    int total_samples = inputs.size();

    // Forward pass for this sample
    conv1.forward(inputs);
    conv1_activation.forward(conv1.output_maps);
    pool1.forward(conv1.output_maps);
    conv2.forward(pool1.output_maps);
    conv2_activation.forward(conv2.output_maps);
    pool2.forward(conv2.output_maps);
    conv3.forward(pool2.output_maps);
    conv3_activation.forward(conv3.output_maps);
    pool3.forward(conv3.output_maps);

    MatrixXd X = pool3.flatten();
    dense1.forward(X);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    activation2.forward(dense2.output);
    dense3.forward(activation2.output);

    // Apply softmax manually to get probabilities
    MatrixXd logits = dense3.output;
    MatrixXd exp_logits = logits.array().exp();
    VectorXd row_sum = exp_logits.rowwise().sum();
    MatrixXd output_probs = exp_logits;
    for (int i = 0; i < output_probs.rows(); i++)
    {
        output_probs.row(i) = output_probs.row(i) / row_sum(i);
    }
    for (int i = 0; i < total_samples; i++)
    {
        // Trouver la classe prédite pour l'échantillon i
        int predicted_class = 0;
        double max_prob = output_probs(i, 0);
        for (int j = 1; j < output_probs.cols(); j++)
        {
            if (output_probs(i, j) > max_prob)
            {
                max_prob = output_probs(i, j);
                predicted_class = j;
            }
        }

        int ground_truth = static_cast<int>(Y[i]);
        if (predicted_class == ground_truth)
        {
            correct_predictions++;
        }

        std::cout << "\nSample " << i + 1 << "/" << total_samples << ":" << std::endl;
        std::cout << "  Predicted class: " << predicted_class
                  << " (" << classes[predicted_class] << ")"
                  << " | Probability: " << max_prob * 100 << "%"
                  << " | Ground truth: " << ground_truth
                  << " (" << classes[ground_truth] << ")"
                  << " | " << (predicted_class == ground_truth ? "CORRECT" : "WRONG") << endl;
    }

    cout << "\n=== RÉSULTATS D'ÉVALUATION ===" << endl;

    // Calculate overall accuracy
    double accuracy = static_cast<double>(correct_predictions) / total_samples * 100.0;
    cout << "\n=== RÉSULTATS FINAUX ===" << endl;
    cout << "Accuracy globale: " << accuracy << "%" << endl;
    cout << "Correct: " << correct_predictions << "/" << total_samples << endl;
    dump_metrics(this->params.iterations, accuracy, correct_predictions, total_samples);
}

Eigen::MatrixXd CNNModel::predict(std::vector<std::vector<MatrixXd>> &inputs)
{
    cout << "\n=== PHASE De Test ===" << endl;

    // Forward pass pour cet echantillon
    conv1.forward(inputs);
    conv1_activation.forward(conv1.output_maps);
    pool1.forward(conv1.output_maps);
    conv2.forward(pool1.output_maps);
    conv2_activation.forward(conv2.output_maps);
    pool2.forward(conv2.output_maps);
    conv3.forward(pool2.output_maps);
    conv3_activation.forward(conv3.output_maps);
    pool3.forward(conv3.output_maps);
    MatrixXd X = pool3.flatten();
    dense1.forward(X);
    activation1.forward(dense1.output);
    dense2.forward(activation1.output);
    activation2.forward(dense2.output);
    dense3.forward(activation2.output);

    return dense3.output;
}

void CNNModel::dump(const std::string &filename)
{
    const std::string PROJECT_WEIGHTS_DIR = "../../db/"; //

    std::cout << "Saving model to: " << filename << std::endl;

    // Créer le dossier parent si nécessaire
    fs::path filepath(PROJECT_WEIGHTS_DIR + filename);
    fs::path dir = filepath.parent_path();

    if (!dir.empty() && !fs::exists(dir))
    {
        std::cout << "Creating directory: " << dir << std::endl;
        if (!fs::create_directories(dir))
        {
            throw std::runtime_error("Failed to create directory: " + dir.string());
        }
    }

    // Ouvrir le fichier
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs)
    {
        throw std::runtime_error("Cannot open file: " + filename +
                                 " (error: " + strerror(errno) + ")");
    }

    // Sauvegarder
    boost::archive::binary_oarchive oa(ofs);
    oa << *this;

    // Vérifier la taille du fichier
    ofs.close();
    if (fs::exists(filepath))
    {
        auto size = fs::file_size(filepath);
        std::cout << "✓ Model saved successfully ("
                  << size << " bytes, "
                  << size / 1024 << " KB)" << std::endl;
    }
    else
    {
        throw std::runtime_error("File was not created: " + filename);
    }
}

bool CNNModel::load(const std::string &filename)
{

    const std::string PROJECT_WEIGHTS_DIR = "../../db/"; //

    fs::path filepath(PROJECT_WEIGHTS_DIR + filename);

    std::cout << "Loading model from: " << filename << std::endl;

    // Vérifier l'existence du fichier
    if (!fs::exists(filepath))
    {
        throw std::runtime_error("File does not exist: " + filename);
    }

    // Vérifier la taille
    auto size = fs::file_size(filepath);
    if (size == 0)
    {
        throw std::runtime_error("File is empty: " + filename);
    }

    std::cout << "File size: " << size << " bytes" << std::endl;

    // Ouvrir et charger
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs)
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    boost::archive::binary_iarchive ia(ifs);
    ia >> *this;

    std::cout << "✓ Model loaded successfully" << std::endl;
    return true;
}

void CNNModel::dump_metrics(int epoch, double loss, double accuracy)
{
    train << "Époque: " << epoch << " | Loss: " << loss
          << " | Accuracy: " << accuracy << "%" << std::endl;
}

void CNNModel::dump_metrics(int epoch, double accuracy, int correct_predictions, int total_samples)
{
    test << "Epoque: " << epoch << " | Accuracy globale: " << accuracy << "%"
         << " | Correct: " << correct_predictions << "/" << total_samples << endl;
}
