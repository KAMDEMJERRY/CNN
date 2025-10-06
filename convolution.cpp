#include "convolution.hpp"

// Implémentation de ConvLayer
ConvLayer::ConvLayer(int in_size, int in_ch, int f_num, int f_size, int pad, int str) 
    : input_size(in_size), input_ch(in_ch), output_ch(f_num), filter_size(f_size), padding(pad), stride(str) 
{
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
    output_maps.resize(output_ch, MatrixXd::Zero(output_size, output_size));
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

void ConvLayer::forward(const std::vector<MatrixXd>& input_maps) {
    if(input_maps.size() != input_ch) {
        throw std::invalid_argument("Le nombre de cartes d'entrée ne correspond pas au nombre de canaux d'entrée.");
    }

    try {
        for(int oc = 0; oc < output_ch; ++oc) {
            output_maps[oc].setZero();
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
                                    sum += input_maps[ic](x, y) * filters[oc][ic](m, n); 
                                }
                            }
                        }
                        output_maps[oc](i, j) += sum;
                    }
                }
            }
            output_maps[oc] = (output_maps[oc].array() + biases(oc)).cwiseMax(0);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de la convolution: " << e.what() << std::endl;
        throw;
    }
}

// Implémentation de PoolLayer
PoolLayer::PoolLayer(int in_size, int in_ch, int p_size) 
    : input_size(in_size), input_ch(in_ch), pool_size(p_size) 
{
    output_size = (input_size + pool_size - 1) / pool_size;
    output_maps.resize(input_ch, MatrixXd::Zero(output_size, output_size));
    flats_output = VectorXd::Zero(output_size * output_size * input_ch);
}

void PoolLayer::forward(const std::vector<MatrixXd>& in_maps) {
    if(in_maps.size() != input_ch) {
        throw std::invalid_argument("Le nombre de cartes d'entrée ne correspond pas au nombre de canaux d'entrée.");
    }

    input_maps = in_maps;
    
    try {
        for(int ic = 0; ic < input_ch; ++ic) {
            for(int i = 0; i < output_size; ++i) {
                for(int j = 0; j < output_size; ++j) {
                    double maxVal = std::numeric_limits<double>::lowest();
                    for(int m = 0; m < pool_size; ++m) {
                        for(int n = 0; n < pool_size; ++n) {
                            int x = i * pool_size + m;
                            int y = j * pool_size + n;
                            if(x < input_maps[ic].rows() && y < input_maps[ic].cols()) {
                                maxVal = std::max(maxVal, input_maps[ic](x, y));
                            }
                        }
                    }
                    output_maps[ic](i, j) = maxVal;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors du pooling: " << e.what() << std::endl;   
        throw;
    }
}

void PoolLayer::flatten() {
    int total_size = output_size * output_size * input_ch;
    flats_output.resize(total_size);
    int index = 0;
    try {
        for(int ic = 0; ic < input_ch; ++ic) {
            for(int i = 0; i < output_size; ++i) {
                for(int j = 0; j < output_size; ++j) {
                    flats_output(index++) = output_maps[ic](i, j);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Erreur lors de l'aplatissement: " << e.what() << std::endl;   
        throw;
    }
}