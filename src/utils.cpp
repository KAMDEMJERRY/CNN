#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
MatrixXd one_hot(const VectorXd& y, int num_labels){
    int uniq = 0;
    int n_samples = y.size();
    if(num_labels<=0){
        uniq =  y.maxCoeff() + 1;
    }else{
        uniq = num_labels;
    }

    MatrixXd ary = MatrixXd::Zero(n_samples, uniq);
    for(int i=0; i<y.size(); i++){
        ary(i, static_cast<int>(y(i))) = 1;
    }
    return ary;
}


void logCNNArchitecture(const ImageDataset& imgDataset, 
                              const ConvLayer& conv1, const PoolLayer& pool1,
                              const ConvLayer& conv2, const PoolLayer& pool2,
                              int image_size, int input_channels, int n_images,
                              const vector<int>& dense_architecture ) {
    
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
        {"Pool2", "(pool_size=" + to_string(pool2.pool_size) + ")"}
    };
    
    vector<tuple<int, int, int>> dimensions = {
        {image_size, image_size, input_channels},
        {conv1.output_size, conv1.output_size, conv1.output_ch},
        {pool1.output_size, pool1.output_size, pool1.input_ch},
        {conv2.output_size, conv2.output_size, conv2.output_ch},
        {pool2.output_size, pool2.output_size, pool2.input_ch}
    };
    
    for (size_t i = 0; i < conv_layers.size(); ++i) {
        auto [h, w, c] = dimensions[i];
        auto [name, info] = conv_layers[i];
        auto [h_next, w_next, c_next] = dimensions[i+1];
        
        cout << (i == 0 ? "┌─ " : "├─ ") << name << ": " 
             << h << "x" << w << "x" << c << " → " 
             << h_next << "x" << w_next << "x" << c_next 
             << " " << info << endl;
    }
    cout << "└─ Flatten: → " << flattened_size << " features" << endl;

    // Partie dense
    cout << "\n--- PARTIE DENSE ---" << endl;
    int total_dense_params = 0;
    for (size_t i = 0; i < full_architecture.size() - 1; ++i) {
        int input_size = full_architecture[i];
        int output_size = full_architecture[i+1];
        int layer_params = input_size * output_size + output_size;
        total_dense_params += layer_params;
        
        string layer_name = (i == full_architecture.size() - 2) ? "Output" : 
                           "Dense" + to_string(i+1);
        string activation = (i == full_architecture.size() - 2) ? "Softmax" : "ReLU";
        
        cout << (i == 0 ? "┌─ " : "├─ ") << layer_name << ": " 
             << input_size << " → " << output_size
             << " | params: " << layer_params 
             << " → " << activation << endl;
    }

    // Résumé
    cout << "\n--- RÉSUMÉ ---" << endl;
    cout << "Architecture: ";
    for (size_t i = 0; i < full_architecture.size(); ++i) {
        cout << full_architecture[i];
        if (i < full_architecture.size() - 1) cout << " → ";
    }
    cout << endl;
    
    cout << "Total paramètres: " << total_dense_params << " (dense only)" << endl;
    cout << "Taille input: " << n_images << " images " << image_size << "x" << image_size << endl;
    cout << "Taille output: " << n_images << " × " << imgDataset.classes.size() << " probabilités" << endl;
}





std::unordered_map<std::string, std::string> loadEnvFile(const std::string& filename = ".env") {
    std::unordered_map<std::string, std::string> env;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
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
void showFilter(const string& layer_name, 
                const vector<vector<MatrixXd>>& filters,
                int cell_size = 50,
                int padding = 5) {
    
    if (filters.empty() || filters[0].empty()) {
        cerr << "Erreur: Aucun filtre à afficher pour " << layer_name << endl;
        return;
    }
    
    // Dimensions des filtres
    int num_output_channels = filters.size();          // nombre de filtres de sortie
    int num_input_channels = filters[0].size();        // nombre de canaux d'entrée
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
    
    for (int out_c = 0; out_c < num_output_channels; out_c++) {
        for (int in_c = 0; in_c < num_input_channels; in_c++) {
            double min_val = filters[out_c][in_c].minCoeff();
            double max_val = filters[out_c][in_c].maxCoeff();
            global_min = std::min(global_min, min_val);
            global_max = std::max(global_max, max_val);
        }
    }
    
    cout << "  Valeurs des filtres: min=" << global_min << ", max=" << global_max << endl;
    
    // Parcourir tous les filtres
    for (int out_c = 0; out_c < num_output_channels; out_c++) {
        for (int in_c = 0; in_c < num_input_channels; in_c++) {
            // Extraire le filtre actuel
            const MatrixXd& filter = filters[out_c][in_c];
            
            // Convertir en Mat OpenCV
            cv::Mat filter_cv(filter_height, filter_width, CV_64FC1);
            for (int i = 0; i < filter_height; i++) {
                for (int j = 0; j < filter_width; j++) {
                    filter_cv.at<double>(i, j) = filter(i, j);
                }
            }
            
            // Normaliser avec les valeurs globales pour une échelle cohérente
            cv::Mat normalized;
            if (global_max > global_min) {
                filter_cv.convertTo(normalized, CV_64FC1, 1.0/(global_max - global_min), 
                                   -global_min/(global_max - global_min));
            } else {
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
    for (int in_c = 0; in_c < num_input_channels; in_c++) {
        int x = padding + in_c * (cell_size + padding) + cell_size/2 - 10;
        cv::putText(big_image, "In" + to_string(in_c),
                   cv::Point(x, 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
    }
    
    // Labels des canaux de sortie (à gauche)
    for (int out_c = 0; out_c < num_output_channels; out_c++) {
        int y = padding + out_c * (cell_size + padding) + cell_size/2 + 5;
        cv::putText(big_image, "Out" + to_string(out_c),
                   cv::Point(5, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
    }
    
    // Titre de l'image
    string title = layer_name + " (" + to_string(num_output_channels) + 
                   "x" + to_string(num_input_channels) + "x" + 
                   to_string(filter_height) + "x" + to_string(filter_width) + ")";
    
    cv::putText(big_image, title,
               cv::Point(total_width/2 - title.length() * 4, total_height - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
    
    // Afficher l'image
    // cv::imshow(layer_name, big_image);
    
    // Sauvegarder l'image
    string filename =  "filters_" + layer_name + ".png";
    cv::imwrite(filename, big_image);
    
    cout << "  Image sauvegardée: " << filename << endl;
    
    // Attendre un peu pour voir l'image
    cv::waitKey(500);
}


// Visualiser chaque filtre de sortie dans une image séparée
void showFilterIndividualOutputs( const string& layer_name,
                                 const vector<vector<MatrixXd>>& filters,
                                 int cell_size = 50) {
    
    if (filters.empty()) return;
    
    int num_output_channels = filters.size();
    int num_input_channels = filters[0].size();
    int filter_height = filters[0][0].rows();
    int filter_width = filters[0][0].cols();
    
    // Pour chaque filtre de sortie, créer une image avec tous ses canaux d'entrée
    for (int out_c = 0; out_c < num_output_channels; out_c++) {
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
        
        for (int in_c = 0; in_c < num_input_channels; in_c++) {
            double local_min = filters[out_c][in_c].minCoeff();
            double local_max = filters[out_c][in_c].maxCoeff();
            min_val = std::min(min_val, local_min);
            max_val = std::max(max_val, local_max);
        }
        
        // Ajouter tous les canaux d'entrée de ce filtre
        for (int in_c = 0; in_c < num_input_channels; in_c++) {
            const MatrixXd& filter = filters[out_c][in_c];
            
            // Convertir et normaliser
            cv::Mat filter_cv(filter_height, filter_width, CV_64FC1);
            for (int i = 0; i < filter_height; i++) {
                for (int j = 0; j < filter_width; j++) {
                    filter_cv.at<double>(i, j) = filter(i, j);
                }
            }
            
            cv::Mat normalized;
            if (max_val > min_val) {
                filter_cv.convertTo(normalized, CV_64FC1, 1.0/(max_val - min_val), 
                                   -min_val/(max_val - min_val));
            } else {
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
void showFilterEnhanced(const string &timeStamp, const string& layer_name, 
                        const vector<vector<MatrixXd>>& filters,
                        bool use_grayscale,
                        int colormap_type,  // Type de colormap
                        bool normalize_per_filter,     // Normaliser chaque filtre individuellement
                        bool show_values ,              // Afficher les valeurs numériques
                        int cell_size,
                        int padding ) {
    
    if (filters.empty() || filters[0].empty()) {
        cerr << "Erreur: Aucun filtre à afficher pour " << layer_name << endl;
        return;
    }
    
    // Dimensions des filtres
    int num_output_channels = filters.size();
    int num_input_channels = filters[0].size();
    int filter_height = filters[0][0].rows();
    int filter_width = filters[0][0].cols();
    
    // Créer une image de grille
    int grid_cols = num_input_channels;
    int grid_rows = num_output_channels;
    
    int total_width = grid_cols * (cell_size + padding) + padding;
    int total_height = grid_rows * (cell_size + padding) + padding + 40;
    
    cv::Mat big_image;
    if (use_grayscale) {
        big_image = cv::Mat(total_height, total_width, CV_8UC1, cv::Scalar(128));
    } else {
        big_image = cv::Mat(total_height, total_width, CV_8UC3, cv::Scalar(50, 50, 50));
    }
    
    // Parcourir tous les filtres
    for (int out_c = 0; out_c < num_output_channels; out_c++) {
        for (int in_c = 0; in_c < num_input_channels; in_c++) {
            const MatrixXd& filter = filters[out_c][in_c];
            
            // Convertir en Mat OpenCV
            cv::Mat filter_cv(filter_height, filter_width, CV_64FC1);
            for (int i = 0; i < filter_height; i++) {
                for (int j = 0; j < filter_width; j++) {
                    filter_cv.at<double>(i, j) = filter(i, j);
                }
            }
            
            // Normalisation
            cv::Mat normalized;
            if (normalize_per_filter) {
                // Normaliser chaque filtre individuellement
                double min_val = filter.minCoeff();
                double max_val = filter.maxCoeff();
                if (max_val > min_val) {
                    filter_cv.convertTo(normalized, CV_64FC1, 1.0/(max_val - min_val), 
                                       -min_val/(max_val - min_val));
                } else {
                    normalized = filter_cv.clone();
                }
            } else {
                // Normaliser avec valeurs globales
                double global_min = 1e9;
                double global_max = -1e9;
                for (int oc = 0; oc < num_output_channels; oc++) {
                    for (int ic = 0; ic < num_input_channels; ic++) {
                        global_min = std::min(global_min, filters[oc][ic].minCoeff());
                        global_max = std::max(global_max, filters[oc][ic].maxCoeff());
                    }
                }
                if (global_max > global_min) {
                    filter_cv.convertTo(normalized, CV_64FC1, 1.0/(global_max - global_min), 
                                       -global_min/(global_max - global_min));
                } else {
                    normalized = filter_cv.clone();
                }
            }
            
            // Convertir en 8-bit
            cv::Mat filter_8u;
            normalized.convertTo(filter_8u, CV_8UC1, 255.0);
            
            // Redimensionner
            cv::Mat resized_filter;
            cv::resize(filter_8u, resized_filter, cv::Size(cell_size, cell_size), 
                      cv::INTER_NEAREST);
            
            // Préparer l'image finale
            cv::Mat final_filter;
            if (use_grayscale) {
                final_filter = resized_filter.clone();
            } else {
                cv::applyColorMap(resized_filter, final_filter, colormap_type);
            }
            
            // Position
            int x = padding + in_c * (cell_size + padding);
            int y = padding + out_c * (cell_size + padding);
            
            // Copier
            final_filter.copyTo(big_image(cv::Rect(x, y, cell_size, cell_size)));
            
            // Afficher les valeurs numériques si demandé (pour petits filtres)
            if (show_values && filter_height <= 5 && filter_width <= 5) {
                cv::Mat value_display(cell_size, cell_size, final_filter.type(), 
                                     use_grayscale ? cv::Scalar(128) : cv::Scalar(50, 50, 50));
                
                // Calculer la taille de police en fonction de la taille de la cellule
                double font_scale = 0.3 * cell_size / 50.0;
                int thickness = 1;
                
                for (int i = 0; i < filter_height; i++) {
                    for (int j = 0; j < filter_width; j++) {
                        string val_str = to_string(filter(i, j));
                        if (val_str.length() > 5) {
                            val_str = val_str.substr(0, 5);
                        }
                        
                        int text_x = j * (cell_size / filter_width) + 5;
                        int text_y = (i + 1) * (cell_size / filter_height) - 5;
                        
                        cv::putText(value_display, val_str,
                                   cv::Point(text_x, text_y),
                                   cv::FONT_HERSHEY_SIMPLEX, font_scale,
                                   use_grayscale ? cv::Scalar(255) : cv::Scalar(255, 255, 255),
                                   thickness);
                    }
                }
                
                // Superposer l'affichage des valeurs
                cv::addWeighted(final_filter, 0.5, value_display, 0.5, 0, final_filter);
                final_filter.copyTo(big_image(cv::Rect(x, y, cell_size, cell_size)));
            }
            
            // Contour
            cv::rectangle(big_image, cv::Rect(x, y, cell_size, cell_size),
                         use_grayscale ? cv::Scalar(200) : cv::Scalar(200, 200, 200), 1);
        }
    }
    
    // Labels et titre (comme précédemment)
    cv::Scalar text_color = use_grayscale ? cv::Scalar(255) : cv::Scalar(255, 255, 255);
    
    for (int in_c = 0; in_c < num_input_channels; in_c++) {
        int x = padding + in_c * (cell_size + padding) + cell_size/2 - 10;
        cv::putText(big_image, "In" + to_string(in_c),
                   cv::Point(x, 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
    }
    
    for (int out_c = 0; out_c < num_output_channels; out_c++) {
        int y = padding + out_c * (cell_size + padding) + cell_size/2 + 5;
        cv::putText(big_image, "Out" + to_string(out_c),
                   cv::Point(5, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1);
    }
    
    string title = layer_name + " (" + to_string(num_output_channels) + 
                   "x" + to_string(num_input_channels) + "x" + 
                   to_string(filter_height) + "x" + to_string(filter_width) + ")";
    
    cv::putText(big_image, title,
               cv::Point(total_width/2 - title.length() * 4, total_height - 10),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);
    
    // Affichage et sauvegarde
    cv::imshow(layer_name, big_image);
    
    string mode_str = use_grayscale ? "gray" : "color";
    string basepath = "../../db/filters_img/" + timeStamp;
    if(!std::filesystem::exists(basepath))
        std::filesystem::create_directories(basepath);
    string filename = basepath + "/" + layer_name + "_" + mode_str + ".png";
    cv::imwrite(filename, big_image);
    
    cv::waitKey(500);
}