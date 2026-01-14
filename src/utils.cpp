#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
MatrixXd one_hot(const VectorXd &y, int num_labels)
{
    int uniq = 0;
    int n_samples = y.size();
    if (num_labels <= 0)
    {
        uniq = y.maxCoeff() + 1;
    }
    else
    {
        uniq = num_labels;
    }

    MatrixXd ary = MatrixXd::Zero(n_samples, uniq);
    for (int i = 0; i < y.size(); i++)
    {
        ary(i, static_cast<int>(y(i))) = 1;
    }
    return ary;
}

void logCNNArchitecture(const ImageDataset &imgDataset,
                        const ConvLayer &conv1, const PoolLayer &pool1,
                        const ConvLayer &conv2, const PoolLayer &pool2,
                        int image_size, int input_channels, int n_images,
                        const vector<int> &dense_architecture)
{

    cout << "\n=== ARCHITECTURE COMPLÈTE DU CNN ===" << endl;
    cout << "======================================" << endl;

    int flattened_size = pool2.output_size * pool2.output_size * pool2.input_ch;

    // Construction de l'architecture dense dynamiquement
    vector<int> full_architecture;
    full_architecture.push_back(flattened_size);
    full_architecture.insert(full_architecture.end(), dense_architecture.begin(), dense_architecture.end());
    full_architecture.push_back(imgDataset.classes.size());

    // Partie convolutionnelle
    cout << "\n--- PARTIE CONVOLUTIONNELLE ---" << endl;
    cout << "Input: " << image_size << "x" << image_size << "x" << input_channels << endl;

    vector<pair<string, string>> conv_layers = {
        {"Conv1", "(" + to_string(conv1.filter_size) + "x" + to_string(conv1.filter_size) +
                      ", filters=" + to_string(conv1.output_ch) + ")"},
        {"Pool1", "(pool_size=" + to_string(pool1.pool_size) + ")"},
        {"Conv2", "(" + to_string(conv2.filter_size) + "x" + to_string(conv2.filter_size) +
                      ", filters=" + to_string(conv2.output_ch) + ")"},
        {"Pool2", "(pool_size=" + to_string(pool2.pool_size) + ")"}};

    vector<tuple<int, int, int>> dimensions = {
        {image_size, image_size, input_channels},
        {conv1.output_size, conv1.output_size, conv1.output_ch},
        {pool1.output_size, pool1.output_size, pool1.input_ch},
        {conv2.output_size, conv2.output_size, conv2.output_ch},
        {pool2.output_size, pool2.output_size, pool2.input_ch}};

    for (size_t i = 0; i < conv_layers.size(); ++i)
    {
        auto [h, w, c] = dimensions[i];
        auto [name, info] = conv_layers[i];
        auto [h_next, w_next, c_next] = dimensions[i + 1];

        cout << (i == 0 ? "┌─ " : "├─ ") << name << ": "
             << h << "x" << w << "x" << c << " → "
             << h_next << "x" << w_next << "x" << c_next
             << " " << info << endl;
    }
    cout << "└─ Flatten: → " << flattened_size << " features" << endl;

    // Partie dense
    cout << "\n--- PARTIE DENSE ---" << endl;
    int total_dense_params = 0;
    for (size_t i = 0; i < full_architecture.size() - 1; ++i)
    {
        int input_size = full_architecture[i];
        int output_size = full_architecture[i + 1];
        int layer_params = input_size * output_size + output_size;
        total_dense_params += layer_params;

        string layer_name = (i == full_architecture.size() - 2) ? "Output" : "Dense" + to_string(i + 1);
        string activation = (i == full_architecture.size() - 2) ? "Softmax" : "ReLU";

        cout << (i == 0 ? "┌─ " : "├─ ") << layer_name << ": "
             << input_size << " → " << output_size
             << " | params: " << layer_params
             << " → " << activation << endl;
    }

    // Résumé
    cout << "\n--- RÉSUMÉ ---" << endl;
    cout << "Architecture: ";
    for (size_t i = 0; i < full_architecture.size(); ++i)
    {
        cout << full_architecture[i];
        if (i < full_architecture.size() - 1)
            cout << " → ";
    }
    cout << endl;

    cout << "Total paramètres: " << total_dense_params << " (dense only)" << endl;
    cout << "Taille input: " << n_images << " images " << image_size << "x" << image_size << endl;
    cout << "Taille output: " << n_images << " × " << imgDataset.classes.size() << " probabilités" << endl;
}

std::unordered_map<std::string, std::string> loadEnvFile(const std::string &filename = ".env")
{
    std::unordered_map<std::string, std::string> env;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        size_t pos = line.find('=');
        if (pos != std::string::npos)
        {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            // Nettoyer les espaces
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            env[key] = value;
        }
    }

    return env;
}

// Utilisation
// int main() {
//     auto env = loadEnvFile();

//     for (const auto& [key, value] : env) {
//         std::cout << key << " = " << value << std::endl;
//     }

//     return 0;
// }

// Fonction pour normaliser et afficher les filtres
void showFilter(const string &layer_name,
                const vector<vector<MatrixXd>> &filters,
                int cell_size = 50,
                int padding = 5)
{

    if (filters.empty() || filters[0].empty())
    {
        cerr << "Erreur: Aucun filtre à afficher pour " << layer_name << endl;
        return;
    }

    // Dimensions des filtres
    int num_output_channels = filters.size();   // nombre de filtres de sortie
    int num_input_channels = filters[0].size(); // nombre de canaux d'entrée
    int filter_height = filters[0][0].rows();
    int filter_width = filters[0][0].cols();

    cout << "Visualisation " << layer_name << ":" << endl;
    cout << "  Output channels: " << num_output_channels << endl;
    cout << "  Input channels: " << num_input_channels << endl;
    cout << "  Filter size: " << filter_height << "x" << filter_width << endl;

    // Créer une image de grille
    int grid_cols = num_input_channels;
    int grid_rows = num_output_channels;

    int total_width = grid_cols * (cell_size + padding) + padding;
    int total_height = grid_rows * (cell_size + padding) + padding + 40; // +40 pour le titre

    cv::Mat big_image(total_height, total_width, CV_8UC3, cv::Scalar(50, 50, 50));

    // Normaliser tous les filtres pour avoir une échelle cohérente
    double global_min = 1e9;
    double global_max = -1e9;

    for (int out_c = 0; out_c < num_output_channels; out_c++)
    {
        for (int in_c = 0; in_c < num_input_channels; in_c++)
        {
            double min_val = filters[out_c][in_c].minCoeff();
            double max_val = filters[out_c][in_c].maxCoeff();
            global_min = std::min(global_min, min_val);
            global_max = std::max(global_max, max_val);
        }
    }

    cout << "  Valeurs des filtres: min=" << global_min << ", max=" << global_max << endl;

    // Parcourir tous les filtres
    for (int out_c = 0; out_c < num_output_channels; out_c++)
    {
        for (int in_c = 0; in_c < num_input_channels; in_c++)
        {
            // Extraire le filtre actuel
            const MatrixXd &filter = filters[out_c][in_c];

            // Convertir en Mat OpenCV
            cv::Mat filter_cv(filter_height, filter_width, CV_64FC1);
            for (int i = 0; i < filter_height; i++)
            {
                for (int j = 0; j < filter_width; j++)
                {
                    filter_cv.at<double>(i, j) = filter(i, j);
                }
            }

            // Normaliser avec les valeurs globales pour une échelle cohérente
            cv::Mat normalized;
            if (global_max > global_min)
            {
                filter_cv.convertTo(normalized, CV_64FC1, 1.0 / (global_max - global_min),
                                    -global_min / (global_max - global_min));
            }
            else
            {
                normalized = filter_cv.clone();
            }

            // Convertir en 8-bit pour l'affichage
            cv::Mat filter_8u;
            normalized.convertTo(filter_8u, CV_8UC1, 255.0);

            // Redimensionner pour l'affichage
            cv::Mat resized_filter;
            cv::resize(filter_8u, resized_filter, cv::Size(cell_size, cell_size),
                       0, 0, cv::INTER_NEAREST);

            // Appliquer une colormap pour mieux voir les variations
            cv::Mat colored_filter;
            cv::applyColorMap(resized_filter, colored_filter, cv::COLORMAP_JET);

            // Position dans la grille
            int x = padding + in_c * (cell_size + padding);
            int y = padding + out_c * (cell_size + padding);

            // Copier dans l'image principale
            colored_filter.copyTo(big_image(cv::Rect(x, y, cell_size, cell_size)));

            // Ajouter un contour pour séparer les cellules
            cv::rectangle(big_image,
                          cv::Rect(x, y, cell_size, cell_size),
                          cv::Scalar(200, 200, 200), 1);
        }
    }

    // Ajouter les labels des axes
    cv::Scalar text_color(255, 255, 255);

    // Labels des canaux d'entrée (en haut)
    for (int in_c = 0; in_c < num_input_channels; in_c++)
    {
        int x = padding + in_c * (cell_size + padding) + cell_size / 2 - 10;
        cv::putText(big_image, "In" + to_string(in_c),
                    cv::Point(x, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
    }

    // Labels des canaux de sortie (à gauche)
    for (int out_c = 0; out_c < num_output_channels; out_c++)
    {
        int y = padding + out_c * (cell_size + padding) + cell_size / 2 + 5;
        cv::putText(big_image, "Out" + to_string(out_c),
                    cv::Point(5, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
    }

    // Titre de l'image
    string title = layer_name + " (" + to_string(num_output_channels) +
                   "x" + to_string(num_input_channels) + "x" +
                   to_string(filter_height) + "x" + to_string(filter_width) + ")";

    cv::putText(big_image, title,
                cv::Point(total_width / 2 - title.length() * 4, total_height - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);

    // Afficher l'image
    // cv::imshow(layer_name, big_image);

    // Sauvegarder l'image
    string filename = "filters_" + layer_name + ".png";
    cv::imwrite(filename, big_image);

    cout << "  Image sauvegardée: " << filename << endl;

    // Attendre un peu pour voir l'image
    cv::waitKey(500);
}

// Visualiser chaque filtre de sortie dans une image séparée
void showFilterIndividualOutputs(const string &layer_name,
                                 const vector<vector<MatrixXd>> &filters,
                                 int cell_size = 50)
{

    if (filters.empty())
        return;

    int num_output_channels = filters.size();
    int num_input_channels = filters[0].size();
    int filter_height = filters[0][0].rows();
    int filter_width = filters[0][0].cols();

    // Pour chaque filtre de sortie, créer une image avec tous ses canaux d'entrée
    for (int out_c = 0; out_c < num_output_channels; out_c++)
    {
        // Déterminer la disposition en grille
        int grid_cols = ceil(sqrt(num_input_channels));
        int grid_rows = ceil(num_input_channels / (double)grid_cols);

        int padding = 5;
        int total_width = grid_cols * (cell_size + padding) + padding;
        int total_height = grid_rows * (cell_size + padding) + padding + 30;

        cv::Mat output_image(total_height, total_width, CV_8UC3, cv::Scalar(50, 50, 50));

        // Normaliser ce filtre spécifique
        double min_val = 1e9;
        double max_val = -1e9;

        for (int in_c = 0; in_c < num_input_channels; in_c++)
        {
            double local_min = filters[out_c][in_c].minCoeff();
            double local_max = filters[out_c][in_c].maxCoeff();
            min_val = std::min(min_val, local_min);
            max_val = std::max(max_val, local_max);
        }

        // Ajouter tous les canaux d'entrée de ce filtre
        for (int in_c = 0; in_c < num_input_channels; in_c++)
        {
            const MatrixXd &filter = filters[out_c][in_c];

            // Convertir et normaliser
            cv::Mat filter_cv(filter_height, filter_width, CV_64FC1);
            for (int i = 0; i < filter_height; i++)
            {
                for (int j = 0; j < filter_width; j++)
                {
                    filter_cv.at<double>(i, j) = filter(i, j);
                }
            }

            cv::Mat normalized;
            if (max_val > min_val)
            {
                filter_cv.convertTo(normalized, CV_64FC1, 1.0 / (max_val - min_val),
                                    -min_val / (max_val - min_val));
            }
            else
            {
                normalized = filter_cv.clone();
            }

            cv::Mat filter_8u;
            normalized.convertTo(filter_8u, CV_8UC1, 255.0);

            cv::Mat resized;
            cv::resize(filter_8u, resized, cv::Size(cell_size, cell_size),
                       cv::INTER_NEAREST);

            // Position dans la grille
            int row = in_c / grid_cols;
            int col = in_c % grid_cols;
            int x = padding + col * (cell_size + padding);
            int y = padding + row * (cell_size + padding);

            // Appliquer colormap et copier
            cv::Mat colored;
            cv::applyColorMap(resized, colored, cv::COLORMAP_VIRIDIS);
            colored.copyTo(output_image(cv::Rect(x, y, cell_size, cell_size)));

            // Label du canal d'entrée
            cv::putText(output_image, "C" + to_string(in_c),
                        cv::Point(x + 5, y + 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
        }

        // Titre
        string title = layer_name + " - Filter Out" + to_string(out_c);
        cv::putText(output_image, title,
                    cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        // Sauvegarder
        string filename = layer_name + "_filter_out" + to_string(out_c) + ".png";
        cv::imwrite(filename, output_image);

        cout << "  Sauvegardé: " << filename << endl;
    }
}


// Version étendue avec plus d'options
void showFilterEnhanced(const string &timeStamp, const string &layer_name,
                        const vector<vector<MatrixXd>> &filters,
                        bool use_grayscale,
                        int colormap_type,
                        bool normalize_per_filter,
                        bool show_values,
                        int cell_size,
                        int padding,
                        bool interpolate)  // NOUVEAU PARAMÈTRE
{

    if (filters.empty() || filters[0].empty())
    {
        cerr << "Erreur: Aucun filtre à afficher pour " << layer_name << endl;
        return;
    }

    // Dimensions des filtres
    int num_output_channels = filters.size();   // Nombre de filtres
    int num_input_channels = filters[0].size(); // Canaux d'entrée
    int filter_height = filters[0][0].rows();
    int filter_width = filters[0][0].cols();

    cout << "Layer: " << layer_name << endl;
    cout << "  Filtres: " << num_output_channels << endl;
    cout << "  Canaux d'entrée: " << num_input_channels << endl;
    cout << "  Taille filtre: " << filter_height << "x" << filter_width << endl;

    // VERSION 1: Affichage par filtre (comme visualize_filters_by_channel)
    // Grille: 1 colonne = 1 filtre, lignes = canaux d'entrée + 1 pour la moyenne

    int grid_cols = num_output_channels;    // 1 colonne par filtre
    int grid_rows = num_input_channels + 1; // Canaux + moyenne

    // Ajuster cell_size si interpolation est désactivée
    if (!interpolate) {
        // Chaque pixel du filtre devient un bloc de pixels
        int pixels_per_value = 15;  // Taille d'un bloc pour une valeur du filtre
        cell_size = std::max(filter_height, filter_width) * pixels_per_value;
    }

    int total_width = grid_cols * (cell_size + padding) + padding;
    int total_height = grid_rows * (cell_size + padding) + padding + 80; // Plus d'espace pour titres

    cv::Mat big_image;
    if (use_grayscale)
    {
        big_image = cv::Mat(total_height, total_width, CV_8UC1, cv::Scalar(128));
    }
    else
    {
        big_image = cv::Mat(total_height, total_width, CV_8UC3, cv::Scalar(50, 50, 50));
    }

    // Calculer min/max global pour la normalisation cohérente
    double global_min = 1e9;
    double global_max = -1e9;
    for (int out_c = 0; out_c < num_output_channels; out_c++)
    {
        for (int in_c = 0; in_c < num_input_channels; in_c++)
        {
            global_min = std::min(global_min, filters[out_c][in_c].minCoeff());
            global_max = std::max(global_max, filters[out_c][in_c].maxCoeff());
        }
    }

    // Parcourir tous les filtres (1 colonne = 1 filtre)
    for (int filter_idx = 0; filter_idx < num_output_channels; filter_idx++)
    {
        // 1. Afficher chaque canal d'entrée du filtre
        for (int channel_idx = 0; channel_idx < num_input_channels; channel_idx++)
        {
            const MatrixXd &filter = filters[filter_idx][channel_idx];

            cv::Mat final_filter;
            
            if (interpolate) {
                // ANCIENNE MÉTHODE AVEC REDIMENSIONNEMENT
                cv::Mat filter_cv(filter_height, filter_width, CV_64FC1);
                for (int i = 0; i < filter_height; i++)
                {
                    for (int j = 0; j < filter_width; j++)
                    {
                        filter_cv.at<double>(i, j) = filter(i, j);
                    }
                }

                // Normaliser
                cv::Mat normalized;
                if (normalize_per_filter)
                {
                    double min_val = filter.minCoeff();
                    double max_val = filter.maxCoeff();
                    if (max_val > min_val)
                    {
                        filter_cv.convertTo(normalized, CV_64FC1, 1.0 / (max_val - min_val),
                                            -min_val / (max_val - min_val));
                    }
                    else
                    {
                        normalized = filter_cv.clone();
                    }
                }
                else
                {
                    if (global_max > global_min)
                    {
                        filter_cv.convertTo(normalized, CV_64FC1, 1.0 / (global_max - global_min),
                                            -global_min / (global_max - global_min));
                    }
                    else
                    {
                        normalized = filter_cv.clone();
                    }
                }

                // Convertir en 8-bit
                cv::Mat filter_8u;
                normalized.convertTo(filter_8u, CV_8UC1, 255.0);

                // Redimensionner AVEC interpolation choisie
                cv::Mat resized_filter;
                int interpolation_method = cv::INTER_NEAREST;  // Peut être changé
                if (cell_size > filter_height * 10) {
                    interpolation_method = cv::INTER_LINEAR;  // Pour gros agrandissements
                }
                
                cv::resize(filter_8u, resized_filter, cv::Size(cell_size, cell_size),
                           interpolation_method);

                // Préparer l'image finale
                if (use_grayscale)
                {
                    final_filter = resized_filter.clone();
                }
                else
                {
                    // Utiliser RdBu pour les valeurs négatives/positives, sinon viridis
                    if (filter.minCoeff() < 0 && filter.maxCoeff() > 0)
                    {
                        cv::applyColorMap(resized_filter, final_filter, cv::COLORMAP_JET);
                    }
                    else
                    {
                        cv::applyColorMap(resized_filter, final_filter, cv::COLORMAP_VIRIDIS);
                    }
                }
            } else {
                // NOUVELLE MÉTHODE SANS INTERPOLATION - grille de pixels
                int pixels_per_value = cell_size / std::max(filter_height, filter_width);
                int actual_cell_size = std::max(filter_height, filter_width) * pixels_per_value;
                
                final_filter = cv::Mat::zeros(actual_cell_size, actual_cell_size, 
                                            use_grayscale ? CV_8UC1 : CV_8UC3);
                
                // Normaliser les valeurs pour l'affichage
                double display_min = normalize_per_filter ? filter.minCoeff() : global_min;
                double display_max = normalize_per_filter ? filter.maxCoeff() : global_max;
                
                for (int i = 0; i < filter_height; i++)
                {
                    for (int j = 0; j < filter_width; j++)
                    {
                        double val = filter(i, j);
                        
                        // Normaliser la valeur
                        double normalized_val;
                        if (display_max > display_min) {
                            normalized_val = (val - display_min) / (display_max - display_min);
                        } else {
                            normalized_val = 0.5;
                        }
                        
                        // Convertir en niveau de gris ou couleur
                        uchar intensity = static_cast<uchar>(normalized_val * 255);
                        
                        // Dessiner un bloc pour cette valeur
                        int block_x = j * pixels_per_value;
                        int block_y = i * pixels_per_value;
                        
                        cv::Rect block(block_x, block_y, pixels_per_value, pixels_per_value);
                        
                        if (use_grayscale) {
                            cv::rectangle(final_filter, block, cv::Scalar(intensity), -1);
                        } else {
                            // Appliquer une colormap manuellement
                            cv::Scalar color;
                            if (filter.minCoeff() < 0 && filter.maxCoeff() > 0) {
                                // Colormap type "RdBu" - bleu pour négatif, rouge pour positif
                                if (val < 0) {
                                    // Bleu: intensité proportionnelle à la valeur négative
                                    int blue_intensity = static_cast<int>(-normalized_val * 255);
                                    color = cv::Scalar(255, 0, blue_intensity); // BGR
                                } else {
                                    // Rouge: intensité proportionnelle à la valeur positive
                                    int red_intensity = static_cast<int>(normalized_val * 255);
                                    color = cv::Scalar(0, 0, red_intensity); // BGR
                                }
                            } else {
                                // Viridis-like colormap
                                if (normalized_val < 0.25) {
                                    color = cv::Scalar(128 + normalized_val * 127, 
                                                        normalized_val * 255, 
                                                        0);
                                } else if (normalized_val < 0.5) {
                                    color = cv::Scalar(255 - (normalized_val-0.25)*255, 
                                                       255, 
                                                       (normalized_val-0.25)*255);
                                } else if (normalized_val < 0.75) {
                                    color = cv::Scalar(0, 
                                                       255 - (normalized_val-0.5)*255, 
                                                       255);
                                } else {
                                    color = cv::Scalar((normalized_val-0.75)*255, 
                                                       0, 
                                                       255);
                                }
                            }
                            cv::rectangle(final_filter, block, color, -1);
                        }
                        
                        // Ajouter une grille
                        cv::rectangle(final_filter, block, 
                                    use_grayscale ? cv::Scalar(100) : cv::Scalar(50, 50, 50), 1);
                        
                        // Afficher la valeur si demandé et si assez de place
                        if (show_values && pixels_per_value >= 20) {
                            string val_str = to_string(val).substr(0, 5);
                            cv::putText(final_filter, val_str,
                                       cv::Point(block_x + 2, block_y + pixels_per_value/2),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.3,
                                       use_grayscale ? cv::Scalar(255) : cv::Scalar(255, 255, 255), 1);
                        }
                    }
                }
                
                // Redimensionner à la taille exacte si nécessaire
                if (final_filter.cols != cell_size || final_filter.rows != cell_size) {
                    cv::resize(final_filter, final_filter, cv::Size(cell_size, cell_size),
                              cv::INTER_NEAREST);
                }
            }

            // Position: 1 colonne par filtre
            int x = padding + filter_idx * (cell_size + padding);
            int y = padding + channel_idx * (cell_size + padding);

            // Copier
            final_filter.copyTo(big_image(cv::Rect(x, y, cell_size, cell_size)));

            // Contour
            cv::rectangle(big_image, cv::Rect(x, y, cell_size, cell_size),
                          use_grayscale ? cv::Scalar(200) : cv::Scalar(200, 200, 200), 1);

            // Label du canal (en haut)
            if (filter_idx == 0)
            {
                string label = "Canal " + to_string(channel_idx);
                cv::putText(big_image, label,
                            cv::Point(x + 5, 15),
                            cv::FONT_HERSHEY_SIMPLEX, 0.3,
                            use_grayscale ? cv::Scalar(255) : cv::Scalar(255, 255, 255), 1);
            }
        }

        // 2. Afficher la moyenne sur tous les canaux (dernière ligne)
        cv::Mat filter_mean = cv::Mat::zeros(filter_height, filter_width, CV_64FC1);

        // Calculer la moyenne
        for (int channel_idx = 0; channel_idx < num_input_channels; channel_idx++)
        {
            const MatrixXd &filter = filters[filter_idx][channel_idx];
            for (int i = 0; i < filter_height; i++)
            {
                for (int j = 0; j < filter_width; j++)
                {
                    filter_mean.at<double>(i, j) += filter(i, j);
                }
            }
        }
        filter_mean /= num_input_channels;

        cv::Mat final_mean;
        if (interpolate) {
            // Ancienne méthode avec resize
            cv::Mat normalized_mean;
            if (global_max > global_min)
            {
                filter_mean.convertTo(normalized_mean, CV_64FC1, 1.0 / (global_max - global_min),
                                      -global_min / (global_max - global_min));
            }
            else
            {
                normalized_mean = filter_mean.clone();
            }

            cv::Mat mean_8u;
            normalized_mean.convertTo(mean_8u, CV_8UC1, 255.0);

            cv::Mat resized_mean;
            cv::resize(mean_8u, resized_mean, cv::Size(cell_size, cell_size),
                       cv::INTER_NEAREST);

            if (use_grayscale)
            {
                final_mean = resized_mean.clone();
            }
            else
            {
                cv::applyColorMap(resized_mean, final_mean, cv::COLORMAP_VIRIDIS);
            }
        } else {
            // Nouvelle méthode sans interpolation
            int pixels_per_value = cell_size / std::max(filter_height, filter_width);
            int actual_cell_size = std::max(filter_height, filter_width) * pixels_per_value;
            
            final_mean = cv::Mat::zeros(actual_cell_size, actual_cell_size, 
                                      use_grayscale ? CV_8UC1 : CV_8UC3);
            
            // Même logique que pour les filtres individuels
            for (int i = 0; i < filter_height; i++)
            {
                for (int j = 0; j < filter_width; j++)
                {
                    double val = filter_mean.at<double>(i, j);
                    double normalized_val;
                    if (global_max > global_min) {
                        normalized_val = (val - global_min) / (global_max - global_min);
                    } else {
                        normalized_val = 0.5;
                    }
                    
                    uchar intensity = static_cast<uchar>(normalized_val * 255);
                    
                    int block_x = j * pixels_per_value;
                    int block_y = i * pixels_per_value;
                    cv::Rect block(block_x, block_y, pixels_per_value, pixels_per_value);
                    
                    if (use_grayscale) {
                        cv::rectangle(final_mean, block, cv::Scalar(intensity), -1);
                    } else {
                        cv::Scalar color;
                        if (val < 0) {
                            int blue_intensity = static_cast<int>(-normalized_val * 255);
                            color = cv::Scalar(255, 0, blue_intensity);
                        } else {
                            int red_intensity = static_cast<int>(normalized_val * 255);
                            color = cv::Scalar(0, 0, red_intensity);
                        }
                        cv::rectangle(final_mean, block, color, -1);
                    }
                    
                    cv::rectangle(final_mean, block, 
                                use_grayscale ? cv::Scalar(100) : cv::Scalar(50, 50, 50), 1);
                }
            }
            
            if (final_mean.cols != cell_size || final_mean.rows != cell_size) {
                cv::resize(final_mean, final_mean, cv::Size(cell_size, cell_size),
                          cv::INTER_NEAREST);
            }
        }

        // Position pour la moyenne (dernière ligne)
        int x = padding + filter_idx * (cell_size + padding);
        int y = padding + num_input_channels * (cell_size + padding);

        final_mean.copyTo(big_image(cv::Rect(x, y, cell_size, cell_size)));

        // Contour rouge pour la moyenne
        cv::rectangle(big_image, cv::Rect(x, y, cell_size, cell_size),
                      use_grayscale ? cv::Scalar(100) : cv::Scalar(0, 0, 255), 2);

        // Label "Moyenne"
        if (filter_idx == 0)
        {
            cv::putText(big_image, "Moyenne",
                        cv::Point(x + 5, y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3,
                        use_grayscale ? cv::Scalar(255) : cv::Scalar(0, 255, 0), 1);
        }

        // 3. Label du filtre (en bas)
        string filter_label = "Filtre " + to_string(filter_idx);
        int label_y = padding + (num_input_channels + 1) * (cell_size + padding) + 15;
        cv::putText(big_image, filter_label,
                    cv::Point(x + cell_size / 2 - 20, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    use_grayscale ? cv::Scalar(255) : cv::Scalar(255, 255, 255), 1);
    }

    // Titre principal
    cv::Scalar text_color = use_grayscale ? cv::Scalar(255) : cv::Scalar(255, 255, 255);
    string title = layer_name + " - " + to_string(num_output_channels) +
                   " filtres x " + to_string(num_input_channels) + " canaux";
    
    if (!interpolate) {
        title += " [Sans interpolation]";
    }

    cv::putText(big_image, title,
                cv::Point(total_width / 2 - title.length() * 4, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1);

    // Légende
    string legend = "Chaque colonne = 1 filtre | Lignes = canaux d'entrée | Dernière ligne = moyenne";
    cv::putText(big_image, legend,
                cv::Point(10, total_height - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1);

    // Statistiques
    string stats = "Min: " + to_string(global_min).substr(0, 6) +
                   " | Max: " + to_string(global_max).substr(0, 6);
    cv::putText(big_image, stats,
                cv::Point(10, total_height - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1);

    // Affichage
    string window_name = layer_name + " - Filters View";
    if (!interpolate) {
        window_name += " [No Interpolation]";
    }
    cv::imshow(window_name, big_image);

    // Sauvegarde
    string basepath = "../../db/filters_img/" + timeStamp;
    if (!std::filesystem::exists(basepath))
    {
        std::filesystem::create_directories(basepath);
    }

    string mode_str = use_grayscale ? "gray" : "color";
    string interp_str = interpolate ? "interp" : "nointerp";
    string filename = basepath + "/" + layer_name + "_" + mode_str + "_" + interp_str + ".png";
    cv::imwrite(filename, big_image);

    // Créer aussi une version miniature pour aperçu rapide
    cv::Mat small_view;
    cv::resize(big_image, small_view, cv::Size(800, 600));
    string small_filename = basepath + "/" + layer_name + "_preview_" + interp_str + ".png";
    cv::imwrite(small_filename, small_view);

    cout << "  Filtres sauvegardés dans: " << filename << endl;
    cout << "  Interpolation: " << (interpolate ? "OUI" : "NON") << endl;

    // Attendre un peu pour visualiser
    cv::waitKey(1000);
}

// Fonction auxiliaire pour la vue traditionnelle (optionnel)
void createTraditionalView(const string &timeStamp, const string &layer_name,
                           const vector<vector<MatrixXd>> &filters,
                           bool use_grayscale,
                           int cell_size, int padding,
                           double global_min, double global_max)
{

    int num_output_channels = filters.size();
    int num_input_channels = filters[0].size();

    // Grille traditionnelle: output_channels x input_channels
    int grid_cols = num_input_channels;
    int grid_rows = num_output_channels;

    int total_width = grid_cols * (cell_size + padding) + padding + 40;
    int total_height = grid_rows * (cell_size + padding) + padding + 40;

    cv::Mat traditional_img;
    if (use_grayscale)
    {
        traditional_img = cv::Mat(total_height, total_width, CV_8UC1, cv::Scalar(128));
    }
    else
    {
        traditional_img = cv::Mat(total_height, total_width, CV_8UC3, cv::Scalar(50, 50, 50));
    }

    // ... (code similaire à votre version originale)
    // Pour gagner de l'espace, je ne répète pas tout le code ici

    string trad_filename = "../../db/filters_img/" + timeStamp + "/" +
                           layer_name + "_traditional.png";
    cv::imwrite(trad_filename, traditional_img);
}
