#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "model.hpp"
#include "eigen_repo.hpp"

namespace boost
{
    namespace serialization
    {

        template <class Archive>
        void serialize(Archive &ar, Activation_ReLU & reLU, const unsigned int version)
        {
            ar & reLU.inputs;
            ar & reLU.output;
            ar & reLU.dinputs;

        }

        template <class Archive>
        void serialize(Archive &ar, Activation_ReLU_Conv & reLU, const unsigned int version)
        {
            ar & reLU.inputs;
            ar & reLU.outputs;
            ar & reLU.dinputs;

        }

        template <class Archive>
        void serialize(Archive &ar, Activation_Softmax & softmax, const unsigned int version)
        {
            ar & softmax.output;
            ar & softmax.dinputs;

        }

        template <class Archive>
        void serialize(Archive &ar, Activation_Softmax_Loss_CategoricalCrossentropy & loss, const unsigned int version){
            ar & loss.activation;
            ar & loss.loss;
            ar & loss.output;
            ar & loss.dinputs;
        }

        template <class Archive>
        void serialize(Archive &ar, DenseLayer &denseLayer, const unsigned int version)
        {
            ar & denseLayer.n_inputs;
            ar & denseLayer.n_neurons;
            ar & denseLayer.weights;
            ar & denseLayer.biases;
            ar & denseLayer.weight_regularizer_l1;
            ar & denseLayer.weight_regularizer_l2;
            ar & denseLayer.bias_regularizer_l1;
            ar & denseLayer.bias_regularizer_l2;
            ar & denseLayer.weights_momentum;
            ar & denseLayer.biases_momentum;
        }

        template <class Archive>
        void serialize(Archive &ar, Optimizer_SGD &sgd, const unsigned int version)
        {
            ar & sgd.learning_rate;
            ar & sgd.current_learning_rate;
            ar & sgd.decay;
            ar & sgd.iterations;
            ar & sgd.momentum;
        }

        template <class Archive>
        void serialize(Archive &ar, ConvLayer &convLayer, const unsigned int version)
        {
            ar & convLayer.input_size;
            ar & convLayer.input_ch;
            ar & convLayer.filter_size;
            ar & convLayer.output_ch;
            ar & convLayer.padding;
            ar & convLayer.stride;
            ar & convLayer.output_size;
            ar & convLayer.weight_regularizer_l1;
            ar & convLayer.weight_regularizer_l2;
            ar & convLayer.bias_regularizer_l1;
            ar & convLayer.bias_regularizer_l2;
            ar & convLayer.filters;
            ar & convLayer.biases;
            ar & convLayer.filters_momentum;
            ar & convLayer.biases_momentum;
        }

        template <class Archive>
        void serialize(Archive &ar, PoolLayer &poolLayer, const unsigned int version)
        {
            ar & poolLayer.input_size;
            ar & poolLayer.input_ch;
            ar & poolLayer.pool_size;
            ar & poolLayer.output_size;
        }

        template <class Archive>
        void serialize(Archive &ar, CNNParameters &CNNparams, const unsigned int version)
        {
            ar & CNNparams.learning_rate;
            ar & CNNparams.decay;
            ar & CNNparams.momentum;
            ar & CNNparams.iterations;
            ar & CNNparams.batch_size;
            ar & CNNparams.epochs;

            ar & CNNparams.d_weight_regularizer_l1;
            ar & CNNparams.d_weight_regularizer_l2;
            ar & CNNparams.d_bias_regularizer_l1;
            ar & CNNparams.d_bias_regularizer_l2;

            ar & CNNparams.c_weight_regularizer_l1;
            ar & CNNparams.c_weight_regularizer_l2;
            ar & CNNparams.c_bias_regularizer_l1;
            ar & CNNparams.c_bias_regularizer_l2;

            ar & CNNparams.checkpoint;

            ar & CNNparams.conv1_inputsize;
            ar & CNNparams.conv1_input_channel_number;
            ar & CNNparams.conv1_filter_number;
            ar & CNNparams.conv1_filter_size;
            ar & CNNparams.conv1_padding;
            ar & CNNparams.conv1_stride;

            ar & CNNparams.pool1_size;

            ar & CNNparams.conv2_inputsize;
            ar & CNNparams.conv2_input_channel_number;
            ar & CNNparams.conv2_filter_number;
            ar & CNNparams.conv2_filter_size;
            ar & CNNparams.conv2_padding;
            ar & CNNparams.conv2_stride;

            ar & CNNparams.pool2_size;

            ar & CNNparams.conv3_inputsize;
            ar & CNNparams.conv3_input_channel_number;
            ar & CNNparams.conv3_filter_number;
            ar & CNNparams.conv3_filter_size;
            ar & CNNparams.conv3_padding;
            ar & CNNparams.conv3_stride;

            ar & CNNparams.pool3_size;

            ar & CNNparams.dense1_inputsize;
            ar & CNNparams.dense2_inputsize; // also number of neuron and  dense2 input size
            ar & CNNparams.dense3_inputsize;
            ar & CNNparams.dense4_inputsize;

            ar & CNNparams.dataset_path;
            ar & CNNparams.databaseURL;
            ar & CNNparams.bd_username;
            ar & CNNparams.bd_password;
        }

        template <class Archive>
        void serialize(Archive &ar, CNNModel &cnnModel, const unsigned int version)
        {
            ar & cnnModel.id;
            ar & cnnModel.params;

            ar & cnnModel.conv1;
            ar & cnnModel.conv1_activation;
            ar & cnnModel.pool1;

            ar & cnnModel.conv2;
            ar & cnnModel.conv2_activation;
            ar & cnnModel.pool2;

            ar & cnnModel.conv3;
            ar & cnnModel.conv3_activation;
            ar & cnnModel.pool3;

            ar & cnnModel.dense1;
            ar & cnnModel.activation1;

            ar & cnnModel.dense2;
            ar & cnnModel.activation2;
            
            ar & cnnModel.dense3;
            // ar & cnnModel.loss_activation;

            ar & cnnModel.optimizer;

            ar & cnnModel.learning_rate;
            ar & cnnModel.decay;
            ar & cnnModel.epochs;
            ar & cnnModel.checkpoint;
            ar & cnnModel.momentum;
            ar & cnnModel.d_weight_regularizer_l1;
            ar & cnnModel.d_weight_regularizer_l2;
            ar & cnnModel.d_bias_regularizer_l1;
            ar & cnnModel.d_bias_regularizer_l2;

            ar & cnnModel.c_weight_regularizer_l1;
            ar & cnnModel.c_weight_regularizer_l2;
            ar & cnnModel.c_bias_regularizer_l1;
            ar & cnnModel.c_bias_regularizer_l2;
        }
    }
}