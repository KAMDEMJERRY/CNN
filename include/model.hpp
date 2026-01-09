#ifndef MODEL
#define MODEL

#include "convolution.hpp"
#include "dense.hpp"
#include "omp.hpp"
#include <fstream>

#define metrics_file  "/home/ndomboukamdem/Documents/INFL/Master 2/Code/log/train.txt"
#define metrics_file1  "/home/ndomboukamdem/Documents/INFL/Master 2/Code/log/eval.txt"
class CNNParameters
{
public:
    double learning_rate;
    double decay;
    double momentum;
    double epochs;
    double iterations = 0;
    double batch_size;

    double d_weight_regularizer_l1 = 0;
    double d_weight_regularizer_l2 = 0;
    double d_bias_regularizer_l1 = 0;
    double d_bias_regularizer_l2 = 0;

    double c_weight_regularizer_l1 = 0;
    double c_weight_regularizer_l2 = 0;
    double c_bias_regularizer_l1 = 0;
    double c_bias_regularizer_l2 = 0;

    int checkpoint;

    // Conv parameters //Square kernels assumed
    int conv1_inputsize;
    int conv1_input_channel_number;
    int conv1_filter_number;
    int conv1_filter_size;
    int conv1_padding;
    int conv1_stride;

    int pool1_size;

    int conv2_inputsize;
    int conv2_input_channel_number;
    int conv2_filter_number;
    int conv2_filter_size;
    int conv2_padding;
    int conv2_stride;

    int pool2_size;

    int conv3_inputsize;
    int conv3_input_channel_number;
    int conv3_filter_number;
    int conv3_filter_size;
    int conv3_padding;
    int conv3_stride;

    int pool3_size;

    // Dense parameters
    int dense1_inputsize;
    int dense2_inputsize; // also number of neuron and  dense2 input size
    int dense3_inputsize;
    int dense4_inputsize;

    // Database
    string dataset_path;
    string databaseURL;
    string bd_username;
    string bd_password;

    CNNParameters() = default;
    ~CNNParameters() = default;
};

class CNNModel
{
public:
    int id;
    CNNParameters params;

    ConvLayer conv1;
    Activation_ReLU_Conv conv1_activation;
    PoolLayer pool1;

    ConvLayer conv2;
    Activation_ReLU_Conv conv2_activation;
    PoolLayer pool2;

    ConvLayer conv3;
    Activation_ReLU_Conv conv3_activation;
    PoolLayer pool3;

    DenseLayer dense1;
    Activation_ReLU activation1;
    DenseLayer dense2;
    Activation_ReLU activation2;
    DenseLayer dense3;
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    Optimizer_SGD optimizer;

    double learning_rate;
    double decay;
    double epochs;
    int checkpoint;
    double momentum;
    double d_weight_regularizer_l1 = 0;
    double d_weight_regularizer_l2 = 0;
    double d_bias_regularizer_l1 = 0;
    double d_bias_regularizer_l2 = 0;

    double c_weight_regularizer_l1 = 0;
    double c_weight_regularizer_l2 = 0;
    double c_bias_regularizer_l1 = 0;
    double c_bias_regularizer_l2 = 0;

    std::ofstream train = std::ofstream(metrics_file, std::ios::app);
    std::ofstream test = std::ofstream(metrics_file1, std::ios::app);

    CNNModel(CNNParameters &params);
    CNNModel() : conv1(), conv2(), conv3(), // You'll need to add default constructors to your layer classes
                 pool1(), pool2(), pool3(),
                 dense1(), dense2(), dense3(),
                 conv1_activation(), conv2_activation(), conv3_activation(),
                 activation1(), activation2(),
                 loss_activation() {};
    ~CNNModel() = default;
    void sethyperparams(CNNParameters &params);
    void compile();
    void fit(std::vector<std::vector<MatrixXd>> &inputs, VectorXd &y);
    void evaluate(std::vector<std::vector<MatrixXd>> &inputs, VectorXd &Y, vector<string> &classes);
    Eigen::MatrixXd predict(std::vector<std::vector<MatrixXd>> &inputs);
    void dump(
        const std::string &filename);
    bool load(
        const std::string &filename);
    void dump_metrics(int epoch, double loss, double accuracy);
    void dump_metrics(int epoch, double accuracy, int correct_predictions, int total_samples);
    // auto forward();
    // auto loss();
    // auto backward();
    // auto update();
};


#endif // MODEL
