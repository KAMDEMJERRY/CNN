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

                output_maps_i[oc] = (output_maps_i[oc].array() + biases(oc)).cwiseMax(0);
            }

        }catch(const std::exception& e){
            std::cerr << "Erreur lors de la convolution:" << e.what() << std::endl;
        }
        
        output_maps.push_back(output_maps_i);
    }
}

// Implémentation de PoolLayer
PoolLayer::PoolLayer(int in_size, int in_ch, int p_size) 
    : input_size(in_size), input_ch(in_ch), pool_size(p_size) 
{
    output_size = (input_size + pool_size - 1) / pool_size;
    // output_maps.resize(input_ch, MatrixXd::Zero(output_size, output_size));
    flats_output = VectorXd::Zero(output_size * output_size * input_ch);
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

vector<vector<MatrixXd>>& PoolLayer::unflatten(MatrixXd& flats){
    int n_in = flats.rows();

    for(int in_i = 0; i_in < n_in; in_i++){

        VectorXd row = MatrixXd.row(in_i);
        
        vector<MatrixXd> dvalue_i(this->input_ch);
        for(int ch=0; ch<input_ch ;ch++){
            dvalue_i[ch] = MatrixXd::Zeros(output_size, output_size);
        }

        int index = 0;
        for(int ch = 0; i=ch < input_ch; ch++){
            for (int i = 0; i<output_size ; i++){
                for (int j = 0; j<output_size; j++){
                    chanels[ch](i, j) = flats(in_i, index++);
                }
            }
        }
        this->dvalue.push_back(dvalue_i);
    }

    return dvalues;

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

