#include "model.hpp"

double calculate_accuracy(const MatrixXd& predictions, const VectorXd& true_labels) {
    int correct = 0;
    int n_samples = predictions.rows();
    
    for (int i = 0; i < n_samples; ++i) {
        // Trouver la classe prédite (indice avec la plus haute probabilité)
        int predicted_class = 0;
        double max_prob = predictions(i, 0);
        for (int j = 1; j < predictions.cols(); ++j) {
            if (predictions(i, j) > max_prob) {
                max_prob = predictions(i, j);
                predicted_class = j;
            }
        }
        
        // Vérifier si la prédiction est correcte
        if (predicted_class == static_cast<int>(true_labels(i))) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / n_samples * 100.0;
}

CNNModel::CNNModel(CNNParameters& params) 
    : params(params),
      conv1(), conv2(), conv3(),  // You'll need to add default constructors to your layer classes
      pool1(), pool2(), pool3(),
      dense1(), dense2(), dense3(),
      conv1_activation(), conv2_activation(), conv3_activation(),
      activation1(), activation2(),
      loss_activation(),
      optimizer(params.learning_rate, params.decay, params.momentum), eval(metrics_file, std::ios::app)

{
    decay = params.decay;
    momentum = params.momentum;
}


CNNModel::CNNModel(){}

void CNNModel::compile() 
{
    conv1 = ConvLayer(params.conv1_inputsize, //conv1.inputsize
                      params.conv1_input_channel_number, // conv1_number of channel of an input
                      params.conv1_filter_number, // conv1.number of filter
                      params.conv1_filter_size, // conv1_size of a filter
                      params.conv1_stride, // conv1_stride
                      params.conv1_padding); // conv1_padding

    pool1 = PoolLayer(conv1.output_size, //pool1_size
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
                      pool2.input_ch,  //input_ch == output_ch
                      params.conv3_filter_number,
                      params.conv3_filter_size,
                      params.conv3_stride,
                      params.conv3_padding);

    pool3 = PoolLayer(conv3.output_size,
                      conv3.output_ch, // because output channels of conv3 = input channels of pool3
                      params.pool3_size);
    
    int input_size = std::pow(pool3.output_size, 2) * pool3.input_ch;
    dense1 = DenseLayer( input_size, params.dense2_inputsize);
    dense2 = DenseLayer( dense1.n_neurons, params.dense3_inputsize);
    dense3 = DenseLayer( dense2.n_neurons, params.dense4_inputsize);

    learning_rate = params.learning_rate;
    momentum = params.momentum;
    decay = params.decay;
    epochs = params.epochs;
    checkpoint = params.checkpoint;
}

void CNNModel::fit(std::vector<std::vector<MatrixXd>>& inputs, VectorXd& y)
{
    
    cout << "Taille d'entrée: " << inputs[0][0].rows() << "x" << inputs[0][0].cols() << endl;

    cout << "\n=== PHASE D'ENTRAÎNEMENT ===" << endl;
    for(int epoch = 0; epoch < epochs; ++epoch){
     
          // Forward pass
            cout << "\nEpoch :" << epoch << "/" << epochs << "\n";
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
            double loss = loss_activation.forward(dense3.output, y);



            // Calcul de la précision toutes les 10 époques
            double accuracy = 0.0;
            if (epoch % params.checkpoint == 0) {
                accuracy = calculate_accuracy(loss_activation.output, y);
            }
            
            // Backward pass
            loss_activation.backward(loss_activation.output, y);
            dense3.backward(loss_activation.dinputs);
            
            activation2.backward(dense3.dinputs);
            dense2.backward(activation2.dinputs);
           
            activation1.backward(dense2.dinputs);
            dense1.backward(activation1.dinputs);

            pool3.backward(pool3.unflatten(dense1.dinputs));
            conv3_activation.backward(pool3.dinput);
            conv3.backward(conv3_activation.dinputs);

            pool2.backward(conv3.dinputs);
            conv2_activation.backward(pool2.dinput);
            conv2.backward(conv2_activation.dinputs);
          
            pool1.backward(conv2.dinputs);
            conv1_activation.backward(pool1.dinput);
            conv1.backward(pool1.dinput);
            
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
            if (epoch % params.checkpoint == 0) {
                cout << "Époque " << epoch 
                     << " | Loss: " << loss 
                     << " | Accuracy: " << accuracy << "%"
                     << " | lr: " << optimizer.current_learning_rate;

                dump_metrics(epoch, loss, accuracy);
            }           
    }
}

void CNNModel::evaluate(std::vector<std::vector<MatrixXd>>& inputs, VectorXd& Y, vector<string>& classes)
{
    cout << "\n=== PHASE D'ÉVALUATION ===" << endl;
    
    int correct_predictions = 0;
    int total_samples = inputs.size();
    
    for(int i = 0; i < total_samples; ++i){
        std::cout << "\n-- Échantillon " << i << "/" << total_samples << std::endl;
        // Extract single input sample
        std::vector<std::vector<MatrixXd>> single_input = {{inputs[i]}};
        
        // Forward pass for this sample
        conv1.forward(single_input);
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
        double sum_exp = exp_logits.sum();
        MatrixXd output_probs = exp_logits / sum_exp;
        
        // Find predicted class
        int predicted_class = 0;
        double max_prob = output_probs(0, 0);
        for(int j = 1; j < output_probs.cols(); ++j) {
            if(output_probs(0, j) > max_prob) {
                max_prob = output_probs(0, j);
                predicted_class = j;
            }
        }
        
        // Get ground truth
        int ground_truth = Y[i];
        
        // Check if prediction is correct
        if(predicted_class == ground_truth) {
            correct_predictions++;
        }
        
        // Log the prediction details
        cout << "Sample " << i << ":" << endl;
        cout << "  Predicted class: " << predicted_class 
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
}

void CNNModel::dump()
{
    std::cout << "Hello world" << std::endl;
}


void CNNModel::dump_metrics(int epoch, double loss, double accuracy){
    eval << "Époque " << epoch << " | Loss: " << loss 
    << " | Accuracy: " << accuracy << "%" << endl;
}