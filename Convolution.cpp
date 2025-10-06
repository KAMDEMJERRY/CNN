#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include "./imgdataset.cpp"
using namespace Eigen;
using namespace std;

class ConvLayer {
public: 
    int input_size;
    int input_ch;
    int filter_size;
    int output_ch;
    int padding;
    int stride;
    int output_size;
   
    std::vector<std::vector<MatrixXd>> filters;
    VectorXd biases;
    std::vector<MatrixXd> output_maps;

    ConvLayer(int in_size, int in_ch, int f_num, int f_size, int pad = 1, int str = 1) 
        : input_size(in_size), input_ch(in_ch), output_ch(f_num), filter_size(f_size), padding(pad), stride(str) 
    {
        // CORRECTION : Calcul correct de la taille de sortie
        output_size = (input_size - filter_size + 2 * padding) / stride + 1;
        
        // CORRECTION : Initialiser output_maps AVANT utilisation
        output_maps.resize(output_ch, MatrixXd::Zero(output_size, output_size));
        
        initialize(); 
    }

    void initialize() {
        // Initialiser les filtres avec des valeurs aléatoires
        filters.resize(output_ch, std::vector<MatrixXd>(input_ch));
        for(int oc = 0; oc < output_ch; ++oc) {
            for(int ic = 0; ic < input_ch; ++ic) {
                filters[oc][ic] = MatrixXd::Random(filter_size, filter_size) * 0.1; // Petites valeurs
            }
        }

        // Initialiser les biais à zéro
        biases = VectorXd::Zero(output_ch);
    }

    void forward(const std::vector<MatrixXd>& input_maps) {
        if(input_maps.size() != input_ch) {
            throw std::invalid_argument("Le nombre de cartes d'entrée ne correspond pas au nombre de canaux d'entrée.");
        }

        try {
            // CORRECTION : Réinitialiser les output_maps à chaque forward
            for(int oc = 0; oc < output_ch; ++oc) {
                output_maps[oc].setZero();
            }

            // Convolution
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
                
                // Ajouter le biais et appliquer ReLU
                output_maps[oc] = (output_maps[oc].array() + biases(oc)).cwiseMax(0); // CORRECTION: cwiseMax au lieu de max
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Erreur lors de la convolution: " << e.what() << std::endl;
            throw;
        }
    }
};

class PoolLayer {
public:
    int input_size;
    int input_ch;
    int pool_size;
    int output_size;

    std::vector<MatrixXd> input_maps;
    std::vector<MatrixXd> output_maps; 
    VectorXd flats_output;
   
    PoolLayer(int in_size, int in_ch, int p_size) 
        : input_size(in_size), input_ch(in_ch), pool_size(p_size) 
    {
        
        output_size = (input_size+ pool_size -1) / pool_size;
        
        // CORRECTION : Initialisation correcte
        output_maps.resize(input_ch, MatrixXd::Zero(output_size, output_size));
        flats_output = VectorXd::Zero(output_size * output_size * input_ch);
    }
   
    void forward(const std::vector<MatrixXd>& in_maps) {
        if(in_maps.size() != input_ch) {
            throw std::invalid_argument("Le nombre de cartes d'entrée ne correspond pas au nombre de canaux d'entrée.");
        }

        input_maps = in_maps;
        
        try {
            // Max Pooling
            for(int ic = 0; ic < input_ch; ++ic) {
                for(int i = 0; i < output_size; ++i) {
                    for(int j = 0; j < output_size; ++j) {
                        double maxVal = std::numeric_limits<double>::lowest(); // CORRECTION: Initialisation correcte
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

    void flatten() {
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
};

// Données d'entrée
int img[5][5] = {
    {3, 0, 1, 2, 7},
    {1, 5, 8, 9, 3},
    {2, 7, 2, 5, 1},
    {0, 1, 3, 1, 7},
    {4, 2, 1, 6, 2}
};

int main() {
    try {
        // Initialisation de l'input
        std::vector<MatrixXd> input_maps(1, MatrixXd(5, 5));
        for(int i = 0; i < 5; ++i) {
            for(int j = 0; j < 5; ++j) {
                input_maps[0](i, j) = img[i][j];
            }
        }
        ImageDataset imgDataset  = loadDataSet();

        cout << "Input map:\n" << input_maps[0] << "\n\n";

        // Création des couches
        ConvLayer conv(5, 1, 2, 3, 1, 1); // 2 filtres de sortie pour tester
        PoolLayer pool(conv.output_size, conv.output_ch, 2);

        ConvLayer conv1(pool.output_size, pool.input_ch, 2, 3, 1, 1); // Nouvelle couche de convolution
        PoolLayer pool1(conv1.output_size, conv1.output_ch, 2);


        // Forward pass
        cout << "=== FORWARD PASS ===" << endl;
        conv.forward(input_maps);
        pool.forward(conv.output_maps);
        conv1.forward(pool.output_maps);
        pool1.forward(conv1.output_maps);
        pool1.flatten();
        
        // Affichage des résultats
        cout << "=== RESULTS ===" << endl;
        cout << "Convolution output size: " << conv1.output_size << "x" << conv1.output_size << endl;
        cout << "Number of output channels: " << conv1.output_ch << endl;
        
        for(int oc = 0; oc < conv1.output_ch; ++oc) {
            cout << "Output map " << oc << ":\n" << conv1.output_maps[oc] << "\n\n";
        }
        
        cout << "Pooled maps (" << pool1.output_size << "x" << pool1.output_size << "):\n";
        for(int oc = 0; oc < pool1.input_ch; ++oc) {
            cout << "Pooled map " << oc << ":\n" << pool1.output_maps[oc] << "\n\n";
        }

        cout << "Flattened output (" << pool1.flats_output.size() << " elements):\n" 
             << pool1.flats_output.transpose() << endl;

    } catch (const std::exception& e) {
        cerr << "ERREUR: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}