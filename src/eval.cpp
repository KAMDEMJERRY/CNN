#include "eval.hpp"
#include "dense.hpp"
#include "matplotlibcpp.h"
#include <omp.h>
#include <ranges>
#include <thread>

template <typename Container>
double mean(const Container &data);

void plot(const std::vector<double> &x, const std::vector<double> &y, const std::string title);

void time_evaluate_convolution(int num_thread)
{

    // int num_thread = 1;
    // if (argc > 1)
    // {
    //     std::istringstream iss(argv[1]);
    //     iss >> num_thread;
    //     cout << "Nbr threads :: (" << num_thread << ")\n";
    // }
    Eigen::setNbThreads(num_thread);

    cout << "=== CHARGEMENT DU DATASET ===" << endl;
    ImageDataset imgDataset(BASE_DATA_PATH, 220, "GRAY"); // RGB or GRAY
    imgDataset.split = 0.8;                               // 80% pour l'entraînement, 20% pour le test
    std::vector<std::vector<MatrixXd>> inputs_train = imgDataset.getTrain().first;
    std::vector<std::vector<MatrixXd>> inputs_test = imgDataset.getTest().first;
    VectorXd y_train = (imgDataset.getTrain().second).cast<double>();
    VectorXd y_test = (imgDataset.getTest()).second.cast<double>();
    int image_size = imgDataset.image_size;   // Les images sont carrées (128x128)
    int input_channels = imgDataset.channels; // Images en niveaux de RGB(3) ou de gris (1)

    // Vérifier que des images ont été chargées
    assert(!imgDataset.images.empty() && "Le dataset d'images ne doit pas être vide");
    assert(!imgDataset.labels.empty() && "Les labels ne doivent pas être vides");

    // Afficher les informations du dataset
    imgDataset.summary();

    // Définir les paramètres du modèle CNN

    CNNParameters params;
    params.epochs = 50;
    params.learning_rate = 0.001;
    params.decay = 1e-4;
    params.momentum = 0.9;
    params.checkpoint = 5; // Corrigé: checkpoints -> checkpoint

    params.d_weight_regularizer_l1 = 0;
    params.d_weight_regularizer_l2 = 1e-4;
    params.d_bias_regularizer_l1 = 0;
    params.d_bias_regularizer_l2 = 1e-4;

    params.c_weight_regularizer_l1 = 0;
    params.c_weight_regularizer_l2 = 1e-4;
    params.c_bias_regularizer_l1 = 0;
    params.c_bias_regularizer_l2 = 1e-4;

    // Configuration Conv1
    params.conv1_inputsize = image_size;
    params.conv1_input_channel_number = input_channels;
    params.conv1_filter_number = 4;
    params.conv1_filter_size = 5;
    params.conv1_padding = 1;
    params.conv1_stride = 1;
    params.pool1_size = 2;

    // Configuration Conv2 (corrigé: input_channel_number devrait être 8, pas input_channels)
    params.conv2_filter_number = 3; // 16;            // 16 filtres comme défini dans conv2
    params.conv2_filter_size = 5;
    params.conv2_padding = 1;
    params.conv2_stride = 1;

    params.pool2_size = 2;

    // Configuration Conv3 (si vous l'utilisez plus tard, sinon vous pouvez supprimer)

    params.conv3_filter_number = 5; // 32;
    params.conv3_filter_size = 3;
    params.conv3_padding = 1;
    params.conv3_stride = 1;

    params.pool3_size = 2;

    // Configuration des couches denses
    params.dense2_inputsize = 20;                        // 64;              // Sortie de dense1 = entrée de dense2
    params.dense3_inputsize = 10;                        // Sortie de dense2 = entrée de dense3
    params.dense4_inputsize = imgDataset.classes.size(); // Sortie de dense3 = nombre de classes

    CNNModel cnn_model(params);
    cnn_model.compile();

    auto begin = std::chrono::high_resolution_clock::now();

    cnn_model.conv1.forward(inputs_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("Time measured: %.3f seconds.\n", elapsed.count() * 1e-9);
}

double time_evaluate_dense(int n_thread)
{
    int N = n_thread;
    int calibre = 500;
    int n = 5000;
    int m = 200;
    int mini_batch_size = static_cast<int>(n / N);

    MatrixXd reproducibleData = generateSyntheticData(n, m, 42); // sample, feature, seed
    DenseLayer dense = DenseLayer(200, 4);                       // n_inputs, n_neurons

    std::vector<double> n_times(calibre);
    std::vector<double> time{};

    std::iota(n_times.begin(), n_times.end(), 1);

    for (auto i : n_times)
    {
        auto begin = std::chrono::high_resolution_clock::now();

#pragma omp parallel for num_threads(n_thread)
        for (int j = 0; j < n; j += mini_batch_size)
        {
            mini_batch_size = std::min(mini_batch_size, n - j);

            auto mini_batch = reproducibleData.block(j, 0, mini_batch_size, m);
            dense.forward(mini_batch);
        }

        auto end = std::chrono::high_resolution_clock::now();
        time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());
    }

    auto elapsed = mean(time);

    std::cout << "Time measured Dense forward : " << elapsed << " milliseconds on " << n_thread << " threads.\n";
    return elapsed;
}

int main(int argc, char *argv[])
{

    std::vector<double> n_thread(8);
    std::vector<double> time{};

    std::iota(n_thread.begin(), n_thread.end(), 1);

    for (auto &n : n_thread)
    {
        time.push_back(time_evaluate_dense(n));
    }

    plot(n_thread, time, "dense_forward_time_per_thread");

    return 0;
}

void plot(const std::vector<double> &x, const std::vector<double> &y, const std::string title)

{
    namespace plt = matplotlibcpp;

    // Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);

    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(x, y);

    // Set x-axis to interval [0,1000000]
    // plt::xlim(0, 1000 * 1000);

    // Add graph title
    plt::title(title);

    // Enable legend.
    plt::legend();
    plt::show();

    // Save the image (file format is determined by the extension)
    std::string base_path = "../../log/";
    std::string file_name = base_path + title + ".png";
    plt::save(file_name);
}

template <typename Container>
double mean(const Container &data)
{
    if (data.empty())
        return 0.0;

    double sum = std::accumulate(std::begin(data), std::end(data), 0.0);
    return sum / data.size();
}