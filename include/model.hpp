#ifndef MODEL
#define MODEL

#include "convolution.hpp"
#include "dense.hpp"

class CNNParameters {
public:
    double learning_rate;
    double epochs;
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

class CNNModel {
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
    double epochs;
    int checkpoint;
    

    CNNModel(CNNParameters& params);   
    CNNModel();   
    ~CNNModel() = default;
    void compile();
    void fit(std::vector<std::vector<MatrixXd>>& inputs, VectorXd& y);
    void evaluate(std::vector<std::vector<MatrixXd>> &inputs, VectorXd &Y, vector<string> &classes);
    void dump(); // every epoch % checkpoint save the model
    void load();
    // auto forward();
    // auto loss();
    // auto backward();
    // auto update();
};


#endif // MODEL