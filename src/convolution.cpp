#include "convolution.hpp"

// Implémentation de ConvLayer
ConvLayer::ConvLayer(int in_size, int in_ch, int f_num, int f_size, int pad, int str) 
    : input_size(in_size), input_ch(in_ch), output_ch(f_num), filter_size(f_size), padding(pad), stride(str) 
{
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
    // output_maps.resize(output_ch, MatrixXd::Zero(output_size, output_size));
    initialize(); 
}

void ConvLayer::initialize() {
    filters.resize(output_ch, std::vector<MatrixXd>(input_ch));
    for(int oc = 0; oc < output_ch; ++oc) {
        for(int ic = 0; ic < input_ch; ++ic) {
            filters[oc][ic] = MatrixXd::Random(filter_size, filter_size) * 0.1;
        }
    }
    biases = VectorXd::Zero(output_ch);
}

void ConvLayer::forward(const std::vector<std::vector<MatrixXd>>& batch_input_maps){
    inputs = batch_input_maps;
    int n_inputs = batch_input_maps.size();
    
    for(int batch_i = 0; batch_i < n_inputs; batch_i++){
        std::vector<MatrixXd> input_maps_i = batch_input_maps[batch_i];
        std::vector<MatrixXd> output_maps_i(output_ch);
        for(int oc = 0; oc < output_ch; oc++){
            output_maps_i[oc] = MatrixXd::Zero(output_size, output_size);
        }

        if(input_maps_i.size() != input_ch){throw std::invalid_argument("Le nombre de cartes d'entree ne correspond pas au nombre de canaux d'entrees");}
        try{
            
            for(int oc = 0; oc < output_ch; ++oc) {
                output_maps_i[oc].setZero();
            }

            for(int oc = 0; oc < output_ch; ++oc) {
                for(int ic = 0; ic < input_ch; ++ic) {
                    for(int i = 0; i < output_size; ++i) {
                    for(int j = 0; j < output_size; ++j) {
                            double sum = 0.0;
                            for(int m = 0; m < filter_size; ++m) {
                            for(int n = 0; n < filter_size; ++n) {
                                    int x = i * stride + m - padding;
                                    int y = j * stride + n - padding;
                                    if(x >= 0 && x < input_size && y >= 0 && y < input_size) {
                                        sum += input_maps_i[ic](x, y) * filters[oc][ic](m, n); 
                                    }
                                }
                            }
                            output_maps_i[oc](i, j) += sum;
                        }
                    }
                }

                output_maps_i[oc] = (output_maps_i[oc].array() + biases(oc)).cwiseMax(0.01 * (output_maps_i[oc].array() + biases(oc)));
            }

        }catch(const std::exception& e){
            std::cerr << "Erreur lors de la convolution:" << e.what() << std::endl;
        }
        
        output_maps.push_back(output_maps_i);
    }
}

// std::vector<std::vector<MatrixXd>>& ConvLayer::backward(const std::vector<std::vector<MatrixXd>>& dvalue){
    
//     std::vector<std::vector<MatrixXd>> dweights;
//     for(int o_ch = 0; o_ch < output_ch; o_ch++){
//         for(int i_ch = 0; i_ch < input_ch; i_ch){
//             for(in_i = 0; in_i < n_in; in_i ++){
//                 dweights[o_ch][i_ch] =  add(dweights[o_ch][i_ch], full_conv(dvalue[in_i][o_ch], inputs[in_i][i_ch]));
//             }
//         }
//     }

//     int n_in = dvalue.size();
//     for(in_i = 0; in_i < n_in; in_i ++){
//         vector<MatrixXd> dinputs_i(input_ch);
//         for(int ch = 0; ch < intput_ch; ch++){
//             dinputs_i[ch] = MatrixXd::Zero(input_size, input_size);
//         }

//         for(int i_ch = 0; i_ch < input_ch; i_ch++){
//             for(int o_ch = 0; o_ch < output_ch; o_ch++){
//                 dinputs_i[ch] = add(dinputs_i[i_ch] , full_conv(filters[o_ch][i_ch], dvalue[o_ch]));
//             }
//         }
       
//         dinputs.push_back(dinputs_i);
//     }
  


    
// }
std::vector<std::vector<MatrixXd>>& ConvLayer::backward(const std::vector<std::vector<MatrixXd>>& dvalue) {
    int n_in = dvalue.size();
    dinputs.clear();
    dweights.clear();
    
    std::cout << "=== BACKWARD DEBUG ===" << std::endl;
    std::cout << "n_in: " << n_in << std::endl;
    std::cout << "input_size: " << input_size << std::endl;
    std::cout << "output_size: " << output_size << std::endl;
    std::cout << "filter_size: " << filter_size << std::endl;
    std::cout << "stride: " << stride << std::endl;
    std::cout << "padding: " << padding << std::endl;
    
    // Vérifier la cohérence des dimensions
    if (n_in > 0 && !dvalue[0].empty()) {
        std::cout << "dvalue[0][0] size: " << dvalue[0][0].rows() << "x" << dvalue[0][0].cols() << std::endl;
        if (dvalue[0][0].rows() != output_size || dvalue[0][0].cols() != output_size) {
            std::cout << "WARNING: dvalue dimensions don't match output_size!" << std::endl;
        }
    }
    
    // 1. Initialiser dweights
    dweights.resize(output_ch);
    for(int o_ch = 0; o_ch < output_ch; o_ch++) {
        dweights[o_ch].resize(input_ch);
        for(int i_ch = 0; i_ch < input_ch; i_ch++) {
            dweights[o_ch][i_ch] = MatrixXd::Zero(filter_size, filter_size);
        }
    }
    
    // 2. Initialiser dbiases
    dbiases = VectorXd::Zero(output_ch);
    
    // 3. Calculer dweights
    for(int o_ch = 0; o_ch < output_ch; o_ch++) {
        for(int i_ch = 0; i_ch < input_ch; i_ch++) {
            for(int in_i = 0; in_i < n_in; in_i++) {
                // Pour dweights: convolution entre input et dvalue
                MatrixXd grad_contrib = conv_for_dweights(inputs[in_i][i_ch], dvalue[in_i][o_ch], filter_size, stride);
                dweights[o_ch][i_ch] += grad_contrib;
            }
        }
    }
    
    // 4. Calculer dbiases
    for(int o_ch = 0; o_ch < output_ch; o_ch++) {
        double bias_grad = 0.0;
        for(int in_i = 0; in_i < n_in; in_i++) {
            bias_grad += dvalue[in_i][o_ch].sum();
        }
        dbiases(o_ch) = bias_grad;
    }
    
    // 5. Calculer dinputs (CONVOLUTION TRANSPOSÉE)
    for(int in_i = 0; in_i < n_in; in_i++) {
        std::vector<MatrixXd> dinputs_i(input_ch);
        for(int i_ch = 0; i_ch < input_ch; i_ch++) {
            dinputs_i[i_ch] = MatrixXd::Zero(input_size, input_size);
        }
        
        for(int i_ch = 0; i_ch < input_ch; i_ch++) {
            for(int o_ch = 0; o_ch < output_ch; o_ch++) {
                // Rotation du filtre de 180 degrés
                MatrixXd rotated_filter = filters[o_ch][i_ch];
                for(int i = 0; i < filter_size/2; i++) {
                    for(int j = 0; j < filter_size; j++) {
                        std::swap(rotated_filter(i, j), rotated_filter(filter_size-1-i, filter_size-1-j));
                    }
                }
                
                // CONVOLUTION TRANSPOSÉE pour dinputs
                MatrixXd grad_contrib = conv_transpose(rotated_filter, dvalue[in_i][o_ch], input_size, stride, padding);
                dinputs_i[i_ch] += grad_contrib;
            }
        }
        
        // Appliquer la dérivée de ReLU
        for(int i_ch = 0; i_ch < input_ch; i_ch++) {
            for(int i = 0; i < input_size; i++) {
                for(int j = 0; j < input_size; j++) {
                    if(inputs[in_i][i_ch](i, j) <= 0) {
                        dinputs_i[i_ch](i, j) = 0;
                    }
                }
            }
        }
        
        dinputs.push_back(dinputs_i);
    }
    
    return dinputs;
}


PoolLayer::PoolLayer(int in_size, int in_ch, int p_size) 
    : input_size(in_size), input_ch(in_ch), pool_size(p_size) 
{
    output_size = (input_size + pool_size - 1) / pool_size;
    // output_maps.resize(input_ch, MatrixXd::Zero(output_size, output_size));
    flats_output = VectorXd::Zero(output_size * output_size * input_ch);
}

vector<vector<MatrixXd>> &PoolLayer::backward(std::vector<std::vector<MatrixXd>> &dvalue)
{
    int n_data = dvalue.size();
    for(int in_i = 0; in_i<n_data; in_i++){
        
        vector<MatrixXd> dinput_i(input_ch);
        for(int ch = 0; ch < input_ch; ch++){
            dinput_i[ch] = MatrixXd::Zero(input_size, input_size);
        }

        for(int ch = 0; ch < input_ch; ch++){
            for(int i = 0; i < output_size; i++){
                for(int j = 0; j< output_size; j++){
                    double maxVal = std::numeric_limits<double>::lowest();
                    vector<int> maxCoord= {0, 0};
                    for(int m = 0; m < pool_size; m++){
                        for(int n = 0; n < pool_size; n++){
                            int x = i * pool_size + n;
                            int y = j * pool_size + m;
                            if(x < dinput_i[ch].rows() && y < dinput_i[ch].cols()){
                                maxVal = std::max(maxVal, dinput_i[ch](x, y));
                                maxCoord[0] = x;
                                maxCoord[1] = y; 
                            }
                        }
                    }
                    dinput_i[ch](maxCoord[0],  maxCoord[1])+= dvalue[in_i][ch](i, j);
                }
            }
        }
        dinput.push_back(dinput_i);
    }
    return dinput;
}



void PoolLayer::forward(const std::vector<std::vector<MatrixXd>>& batch_in_maps){
    int n_inputs = batch_in_maps.size();
    int output_ch = input_ch;
    for(int i_ = 0; i_ < n_inputs; i_++){
        std::vector<MatrixXd> input_maps_i = batch_in_maps[i_];
        std::vector<MatrixXd> output_maps_i(output_ch);
        for(int oc = 0; oc < output_ch; oc++){
            output_maps_i[oc] = MatrixXd::Zero(output_size, output_size);
        }

        if(input_maps_i.size() != input_ch) {throw std::invalid_argument("Le nombre de cartes d'entrée ne correspond pas au nombre de canaux d'entrée."); }
        try {
            for(int ic = 0; ic < input_ch; ++ic) {
                for(int i = 0; i < output_size; ++i) {
                    for(int j = 0; j < output_size; ++j) {
                        double maxVal = std::numeric_limits<double>::lowest();
                        for(int m = 0; m < pool_size; ++m) {
                            for(int n = 0; n < pool_size; ++n) {
                                int x = i * pool_size + m;
                                int y = j * pool_size + n;
                                if(x < input_maps_i[ic].rows() && y < input_maps_i[ic].cols()) {
                                    maxVal = std::max(maxVal, input_maps_i[ic](x, y));
                                }
                            }
                        }
                        output_maps_i[ic](i, j) = maxVal;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Erreur lors du pooling: " << e.what() << std::endl;   
            throw;
        }

        output_maps.push_back(output_maps_i);
    }
}

std::vector<std::vector<MatrixXd>>& PoolLayer::unflatten(MatrixXd& flats) {
    int n_in = flats.rows();
    dvalue.clear(); 
    
    for(int in_i = 0; in_i < n_in; in_i++) {
        VectorXd row = flats.row(in_i);
        
        std::vector<MatrixXd> dvalue_i(input_ch); 
        for(int ch = 0; ch < input_ch; ch++) {
            dvalue_i[ch] = MatrixXd::Zero(output_size, output_size);
        }

        int index = 0;
        for(int ch = 0; ch < input_ch; ch++) {
            for(int i = 0; i < output_size; i++) {
                for(int j = 0; j < output_size; j++) {
                    dvalue_i[ch](i, j) = row(index++);
                }
            }
        }
        dvalue.push_back(dvalue_i);
    }

    return dvalue;
}

MatrixXd& PoolLayer::flatten() {
    int total_size = output_size * output_size * input_ch;
    int n_inputs = output_maps.size();
    
    flats_output.resize(n_inputs, total_size);
    
    for(int i_ = 0; i_ < n_inputs; i_++){
        std::vector<MatrixXd> output_maps_i = output_maps[i_];
        VectorXd flats_output_i;
        flats_output_i.resize(total_size);

        int index = 0;

        try {
            for(int ic = 0; ic < input_ch; ++ic) {
                for(int i = 0; i < output_size; ++i) {
                    for(int j = 0; j < output_size; ++j) {
                        flats_output_i(index++) = output_maps_i[ic](i, j);
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Erreur lors de l'aplatissement: " << e.what() << std::endl;   
            throw;
        }

        flats_output.row(i_) = flats_output_i;
    }

    return flats_output;

}

MatrixXd conv_for_dweights(const MatrixXd& input, const MatrixXd& dvalue, int filter_size, int stride) {
    // Pour dweights: convolution entre input (128x128) et dvalue (128x128)
    // Résultat doit être filter_size x filter_size
    
    MatrixXd result = MatrixXd::Zero(filter_size, filter_size);
    int input_size = input.rows();
    int dvalue_size = dvalue.rows();
    
    // Nous devons "réduire" la convolution pour obtenir filter_size
    // Un moyen simple est de faire une convolution avec un grand stride
    int effective_stride = (input_size - 1) / (filter_size - 1);
    
    for(int i = 0; i < filter_size; i++) {
        for(int j = 0; j < filter_size; j++) {
            double sum = 0.0;
            for(int m = 0; m < dvalue_size; m++) {
                for(int n = 0; n < dvalue_size; n++) {
                    int x = i * effective_stride + m;
                    int y = j * effective_stride + n;
                    if(x < input_size && y < input_size) {
                        sum += input(x, y) * dvalue(m, n);
                    }
                }
            }
            result(i, j) = sum;
        }
    }
    return result;
}

MatrixXd conv_transpose(const MatrixXd& kernel, const MatrixXd& dvalue, int output_size, int stride, int padding) {
    // Convolution transposée pour dinputs
    // kernel: filtre roté (3x3), dvalue: (128x128), output_size: 128
    
    MatrixXd result = MatrixXd::Zero(output_size, output_size);
    int kernel_size = kernel.rows();
    int dvalue_size = dvalue.rows();
    
    // Parcourir chaque position dans dvalue
    for(int i = 0; i < dvalue_size; i++) {
        for(int j = 0; j < dvalue_size; j++) {
            // Pour chaque position dans le kernel
            for(int m = 0; m < kernel_size; m++) {
                for(int n = 0; n < kernel_size; n++) {
                    // Calculer la position correspondante dans l'output
                    int x = i * stride + m - padding;
                    int y = j * stride + n - padding;
                    
                    // Vérifier les limites
                    if(x >= 0 && x < output_size && y >= 0 && y < output_size) {
                        result(x, y) += kernel(m, n) * dvalue(i, j);
                    }
                }
            }
        }
    }
    return result;
}