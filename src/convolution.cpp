#include "convolution.hpp"

// Suppose we are working with square matrices for simplicity
MatrixXd convolution2D(MatrixXd Input, MatrixXd kernel, int padding, int stride)
{

    assert(kernel.rows() == kernel.cols() && "Only square kernels are supported");
    assert(Input.rows() == Input.cols() && "Only square input matrices are supported");

    int o = std::floor((Input.rows() - kernel.rows() + 2 * padding) / stride) + 1;
    MatrixXd Output = MatrixXd::Zero(o, o);

    for (int i = 0; i < o; i++)
    {
        for (int j = 0; j < o; j++)
        {
            double sum = 0.0;
            for (int m = 0; m < kernel.rows(); m++)
            {
                for (int n = 0; n < kernel.cols(); n++)
                {
                    int x = i * stride + m - padding;
                    int y = j * stride + n - padding;
                    if (x >= 0 && x < Input.rows() && y >= 0 && y < Input.cols())
                    {
                        sum += Input(x, y) * kernel(m, n);
                    }
                }
            }
            Output(i, j) = sum;
        }
    }

    return Output;
}

// Implémentation de ConvLayer
ConvLayer::ConvLayer(int in_size, int in_ch, int f_num, int f_size, int pad, int str,
                     double weight_regularizer_l1, double weight_regularizer_l2,
                     double bias_regularizer_l1, double bias_regularizer_l2

                     )
    : input_size(in_size), input_ch(in_ch), output_ch(f_num), filter_size(f_size), padding(pad), stride(str)
{
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;

    this->weight_regularizer_l1 = weight_regularizer_l1;
    this->weight_regularizer_l2 = weight_regularizer_l2;
    this->bias_regularizer_l1 = bias_regularizer_l1;
    this->bias_regularizer_l2 = bias_regularizer_l2;

    // output_maps.resize(output_ch, MatrixXd::Zero(output_size, output_size));
    initialize();
}

void ConvLayer::initialize()
{
    filters.resize(output_ch, std::vector<MatrixXd>(input_ch));
    filters_momentum.resize(output_ch, std::vector<MatrixXd>(input_ch));

    // Calcul du scale pour He initialization
    double scale = sqrt(2.0 / (input_ch * filter_size * filter_size));

    for (int oc = 0; oc < output_ch; ++oc)
    {
        for (int ic = 0; ic < input_ch; ++ic)
        {
            filters[oc][ic] = MatrixXd::Random(filter_size, filter_size) * scale;
            filters_momentum[oc][ic] = MatrixXd::Zero(filter_size, filter_size);
        }
    }

    biases = VectorXd::Zero(output_ch);
    biases_momentum = VectorXd::Zero(output_ch);
}

void ConvLayer::forward(const std::vector<std::vector<MatrixXd>> &batch_input_maps)
{
    inputs = batch_input_maps;
    int n_inputs = batch_input_maps.size();
    output_maps.clear();

    // #pragma omp parallel for
    for (int batch_i = 0; batch_i < n_inputs; batch_i++)
    { // pour chaque input
        const std::vector<MatrixXd> &input_maps_i = batch_input_maps[batch_i];
        std::vector<MatrixXd> output_maps_i(output_ch);

        for (int oc = 0; oc < output_ch; oc++)
        { // On initialize les cartes de caracteristiques de la sortie pour l'input
            output_maps_i[oc] = MatrixXd::Zero(output_size, output_size);
        }

        // std::cout << "\nFrom convforward: input maps for sample i (" << (input_maps_i.size()) << ")\n";
        // std::cout << "\nFrom convforward: Conv inputchannel (" << (input_ch) << ")\n";

        if (input_maps_i.size() != input_ch)
        {
            throw std::invalid_argument("Le nombre de cartes d'entree ne correspond pas au nombre de canaux d'entrees");
        }
        try
        {
            // # pragma omp parallel for collapse(1)
            for (int oc = 0; oc < output_ch; ++oc)
            { // Pour chaque canaux de la sortie de l'input (chaque kernel de la couche de convolution),
                for (int ic = 0; ic < input_ch; ++ic)
                { // pour chaque canaux (de l'entree et d'un kernel)
                    // On realise la convolution 2D
                    for (int i = 0; i < output_size; ++i)
                    {
                        for (int j = 0; j < output_size; ++j)
                        {
                            double sum = 0.0;
                            for (int m = 0; m < filter_size; ++m)
                            {
                                for (int n = 0; n < filter_size; ++n)
                                {
                                    int x = i * stride + m - padding;
                                    int y = j * stride + n - padding;
                                    if (x >= 0 && x < input_size && y >= 0 && y < input_size)
                                    {
                                        sum += input_maps_i[ic](x, y) * filters[oc][ic](m, n);
                                    }
                                }
                            }
                            output_maps_i[oc](i, j) += sum;
                        }
                    }
                }

                output_maps_i[oc] = output_maps_i[oc].array() + biases(oc);
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Erreur lors de la convolution:" << e.what() << std::endl;
        }

        output_maps.push_back(output_maps_i);
    }
}

std::vector<std::vector<MatrixXd>> &ConvLayer::backward(const std::vector<std::vector<MatrixXd>> &dvalue)
{
    int n_in = dvalue.size();
    dinputs.clear();
    dweights.clear();

    // std::cout << "=== BACKWARD DEBUG ===" << std::endl;
    // std::cout << "n_in: " << n_in << std::endl;
    // std::cout << "input_size: " << input_size << std::endl;
    // std::cout << "output_size: " << output_size << std::endl;
    // std::cout << "filter_size: " << filter_size << std::endl;
    // std::cout << "stride: " << stride << std::endl;
    // std::cout << "padding: " << padding << std::endl;

    // Vérifier la cohérence des dimensions
    if (n_in > 0 && !dvalue[0].empty())
    {
        if (dvalue[0][0].rows() != output_size || dvalue[0][0].cols() != output_size)
        {
            std::cout << "WARNING: dvalue dimensions don't match output_size!" << std::endl;
        }
    }

    // 1. Initialiser dweights
    dweights.resize(output_ch);
    for (int o_ch = 0; o_ch < output_ch; o_ch++)
    {
        dweights[o_ch].resize(input_ch);
        for (int i_ch = 0; i_ch < input_ch; i_ch++)
        {
            dweights[o_ch][i_ch] = MatrixXd::Zero(filter_size, filter_size);
        }
    }

    // 2. Initialiser dbiases
    dbiases = VectorXd::Zero(output_ch);

    // 3. Calculer dweights
    for (int in_i = 0; in_i < n_in; in_i++)
    { // pour chaque entre
        for (int o_ch = 0; o_ch < output_ch; o_ch++)
        { // pour chaque carte de sortie de l'entree
            // 4. Calculer dbiases
            dbiases(o_ch) += dvalue[in_i][o_ch].sum();
            for (int i_ch = 0; i_ch < input_ch; i_ch++)
            { // pour chaque carte d'une entree
                // Pour dweights: convolution entre input et dvalue
                for (int m = 0; m < filter_size; m++)
                {
                    for (int n = 0; n < filter_size; n++)
                    {
                        double sum = 0.0;
                        for (int i = 0; i < output_size; i++)
                        {
                            for (int j = 0; j < output_size; j++)
                            {
                                int x = i * stride + m - padding;
                                int y = j * stride + n - padding;
                                if (x >= 0 && x < input_size && y >= 0 && y < input_size)
                                {
                                    sum += inputs[in_i][i_ch](x, y) * dvalue[in_i][o_ch](i, j);
                                }
                            }
                        }
                        dweights[o_ch][i_ch](m, n) += sum;
                    }
                }
            }
        }
    }

    // Gradients sur la regularization
    // L1 sur les filtres
    if (weight_regularizer_l1 > 0)
    {
        std::vector<std::vector<MatrixXd>> dL1;
        dL1.resize(output_ch);
        for (int o_ch = 0; o_ch < output_ch; o_ch++)
        {
            dL1[o_ch].resize(input_ch);
            for (int i_ch = 0; i_ch < input_ch; i_ch++)
            {
                dL1[o_ch][i_ch] = MatrixXd::Ones(filter_size, filter_size);
                dL1[o_ch][i_ch] = (filters[o_ch][i_ch].array() < 0).select(Eigen::MatrixXd::Constant(filters[o_ch][i_ch].rows(), filters[o_ch][i_ch].cols(), -1.), Eigen::MatrixXd::Constant(filters[o_ch][i_ch].rows(), filters[o_ch][i_ch].cols(), 1.));
                dweights[o_ch][i_ch] = dweights[o_ch][i_ch].array() + weight_regularizer_l1 * dL1[o_ch][i_ch].array();
            }
        }
    }

    // L2 sur les filtres
    if (weight_regularizer_l2 > 0)
    {
        for (int o_ch = 0; o_ch < output_ch; o_ch++)
        {
            for (int i_ch = 0; i_ch < input_ch; i_ch++)
            {
                dweights[o_ch][i_ch] = dweights[o_ch][i_ch].array() + 2 * weight_regularizer_l1 * filters[o_ch][i_ch].array();
            }
        }
    }

    // L1 sur les biais
    if (bias_regularizer_l1 > 0)
    {
        VectorXd dL1 = VectorXd::Ones(biases.size());
        dL1 = (biases.array() < 0).select(Eigen::VectorXd::Constant(biases.size(), -1.), Eigen::VectorXd::Constant(biases.size(), 1.));
        dbiases = dbiases.array() + bias_regularizer_l1 * dL1.array();
    }

    // L2 sur les biais
    if (bias_regularizer_l2 > 0)
    {
        dbiases = dbiases.array() + 2 * bias_regularizer_l2 * biases.array();
    }

    // 5. Calculer dinputs (CONVOLUTION TRANSPOSÉE)
    for (int in_i = 0; in_i < n_in; in_i++)
    {
        std::vector<MatrixXd> dinputs_i(input_ch);
        for (int i_ch = 0; i_ch < input_ch; i_ch++)
        {
            dinputs_i[i_ch] = MatrixXd::Zero(input_size, input_size);
        }

        for (int i_ch = 0; i_ch < input_ch; i_ch++)
        {
            for (int o_ch = 0; o_ch < output_ch; o_ch++)
            {
                for (int i = 0; i < output_size; i++)
                {
                    for (int j = 0; j < output_size; j++)
                    {
                        for (int m = 0; m < filter_size; m++)
                        {
                            for (int n = 0; n < filter_size; n++)
                            {
                                int x = i * stride + m - (filter_size - padding - 1);
                                int y = j * stride + n - (filter_size - padding - 1);
                                if (x >= 0 && x < input_size && y >= 0 && y < input_size)
                                {
                                    dinputs_i[i_ch](x, y) += filters[o_ch][i_ch](m, n) * dvalue[in_i][o_ch](i, j);
                                }
                            }
                        }
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
    dinput.clear();

    for (int in_i = 0; in_i < n_data; in_i++)
    {

        vector<MatrixXd> dinput_i(input_ch);
        for (int ch = 0; ch < input_ch; ch++)
        {
            dinput_i[ch] = MatrixXd::Zero(input_size, input_size);
        }

        for (int ch = 0; ch < input_ch; ch++)
        {
            for (int i = 0; i < output_size; i++)
            {
                for (int j = 0; j < output_size; j++)
                {
                    auto maxCoord = max_indices[in_i][ch][i * output_size + j];
                    dinput_i[ch](maxCoord.first, maxCoord.second) += dvalue[in_i][ch](i, j);
                }
            }
        }
        dinput.push_back(dinput_i);
    }
    return dinput;
}

void PoolLayer::forward(const std::vector<std::vector<MatrixXd>> &batch_in_maps)
{
    
    int n_inputs = batch_in_maps.size();
    int output_ch = this->input_ch;
    output_maps.clear();
    max_indices.clear();

    this->input_maps = batch_in_maps;

    for (int i_ = 0; i_ < n_inputs; i_++)
    {
        std::vector<MatrixXd> input_maps_i = batch_in_maps[i_];
        std::vector<MatrixXd> output_maps_i(output_ch);
        std::vector<std::vector<std::pair<int, int>>> max_indices_i(output_ch);

        

        for (int oc = 0; oc < output_ch; oc++)
        {
            output_maps_i[oc] = MatrixXd::Zero(output_size, output_size);
            max_indices_i[oc].resize(output_size * output_size);
        }

        if (input_maps_i.size() != this->input_ch)
        {   
           
       
            throw std::invalid_argument("Le nombre de cartes d'entrée ne correspond pas au nombre de canaux d'entrée.");
        }
        try
        {
            for (int ic = 0; ic < input_ch; ++ic)
            {
                for (int i = 0; i < output_size; ++i)
                {
                    for (int j = 0; j < output_size; ++j)
                    {
                        double maxVal = std::numeric_limits<double>::lowest();
                        int maxX = -1, maxY = -1;
                        for (int m = 0; m < pool_size; ++m)
                        {
                            for (int n = 0; n < pool_size; ++n)
                            {
                                int x = i * pool_size + m;
                                int y = j * pool_size + n;
                                if (x < input_maps_i[ic].rows() && y < input_maps_i[ic].cols())
                                {
                                    if (input_maps_i[ic](x, y) > maxVal)
                                    {
                                        maxVal = input_maps_i[ic](x, y);
                                        maxX = x;
                                        maxY = y;
                                    }
                                }
                            }
                        }
                        output_maps_i[ic](i, j) = maxVal;
                        max_indices_i[ic][i * output_size + j] = std::make_pair(maxX, maxY);
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Erreur lors du pooling: " << e.what() << std::endl;
            throw;
        }
        output_maps.push_back(output_maps_i);
        max_indices.push_back(max_indices_i);
    }
}

std::vector<std::vector<MatrixXd>> &PoolLayer::unflatten(MatrixXd &flats)
{
    int n_in = flats.rows();
    dvalue.clear();

    for (int in_i = 0; in_i < n_in; in_i++)
    {
        VectorXd row = flats.row(in_i);

        std::vector<MatrixXd> dvalue_i(input_ch);
        for (int ch = 0; ch < input_ch; ch++)
        {
            dvalue_i[ch] = MatrixXd::Zero(output_size, output_size);
        }

        int index = 0;
        for (int ch = 0; ch < input_ch; ch++)
        {
            for (int i = 0; i < output_size; i++)
            {
                for (int j = 0; j < output_size; j++)
                {
                    dvalue_i[ch](i, j) = row(index++);
                }
            }
        }
        dvalue.push_back(dvalue_i);
    }

    return dvalue;
}

MatrixXd &PoolLayer::flatten()
{
    int total_size = output_size * output_size * input_ch;
    int n_inputs = output_maps.size();

    flats_output.resize(n_inputs, total_size);

    for (int i_ = 0; i_ < n_inputs; i_++)
    {
        std::vector<MatrixXd> output_maps_i = output_maps[i_];
        VectorXd flats_output_i;
        flats_output_i.resize(total_size);

        int index = 0;

        try
        {
            for (int ic = 0; ic < input_ch; ++ic)
            {
                for (int i = 0; i < output_size; ++i)
                {
                    for (int j = 0; j < output_size; ++j)
                    {
                        flats_output_i(index++) = output_maps_i[ic](i, j);
                    }
                }
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Erreur lors de l'aplatissement: " << e.what() << std::endl;
            throw;
        }

        flats_output.row(i_) = flats_output_i;
    }

    return flats_output;
}

std::vector<std::vector<MatrixXd>> &Activation_ReLU_Conv::forward(const std::vector<std::vector<MatrixXd>> &inputs)
{
    this->inputs = inputs;
    outputs.clear();

    for (const auto &input_i : inputs)
    {
        std::vector<MatrixXd> output_i;
        for (const auto &map : input_i)
        {
            output_i.push_back(map.array().max(0).matrix());
        }
        outputs.push_back(output_i);
    }
    return outputs;
}

std::vector<std::vector<MatrixXd>> &Activation_ReLU_Conv::backward(const std::vector<std::vector<MatrixXd>> &dvalues)
{
    int n_in = dvalues.size();
    int n_ch = dvalues[0].size();
    dinputs.clear();

    for (int i_ = 0; i_ < n_in; i_++)
    {
        std::vector<MatrixXd> dinputs_i;
        for (int ch = 0; ch < n_ch; ch++)
        {
            MatrixXd grad = dvalues[i_][ch].array() * (inputs[i_][ch].array() > 0).cast<double>();
            dinputs_i.push_back(grad);
        }
        dinputs.push_back(dinputs_i);
    }
    return dinputs;
}
