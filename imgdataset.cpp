#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <algorithm>
#include <random>

using namespace Eigen;
using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem;

String BASE_DATA_PATH = "../dataset/bloodcellsub/images/TRAIN/";
vector<String> class_path = {"EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"};


class ImageDataset {
public:
    vector<String> classes;
    vector<MatrixXd> images;
    vector<String> labels;
    vector<int> encoded_labels;

    ImageDataset( 
        vector<String> classes,
        vector<MatrixXd> images,
        vector<String> labels
    ): classes(classes), images(images), labels(labels){};

    vector<int> ordinalEncoding(vector<string>& classes, vector<string>& data_labels){
        for(string& lab : data_labels){
            for(int i = 0; i < classes.size(); i++){
                if(lab == classes[i]){
                    encoded_labels.push_back(i);  // Ajouter l'index de la classe
                    break;  // Sortir de la boucle une fois trouvé
                }
            }
        }
        return encoded_labels;
    }

    vector<MatrixXd> getX(){
        return images;
    }

    vector<string> getY(){
        return labels;
    }

    vector<int> getY_encoded(){
        return encoded_labels;
    }

};

class ImageDatasetLoader {
private:
    vector<MatrixXd> images;
    vector<string> labels;
    int image_height;
    int image_width;

public:
    // Charger une seule image
    MatrixXd loadImage(const string& image_path, int target_height = -1, int target_width = -1) {
        // Charger l'image
        Mat image = imread(image_path, IMREAD_GRAYSCALE);
        
        if (image.empty()) {
            throw runtime_error("Cannot load image: " + image_path);
        }
        
        // Redimensionner si nécessaire
        if (target_height > 0 && target_width > 0) {
            Mat resized_image;
            resize(image, resized_image, Size(target_width, target_height));
            image = resized_image;
        }
        
        // Convertir OpenCV Mat en Eigen Matrix
        MatrixXd eigen_image(image.rows, image.cols);
        
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                eigen_image(i, j) = static_cast<double>(image.at<uchar>(i, j)) / 255.0;
            }
        }
        
        return eigen_image;
    }
    
    // Charger un dataset complet d'images
    void loadDataset(const vector<string>& image_paths, 
                    const vector<string>& image_labels = {},
                    int target_height = 64, 
                    int target_width = 64) {
        
        images.clear();
        labels.clear();
        image_height = target_height;
        image_width = target_width;
        
        for (size_t i = 0; i < image_paths.size(); ++i) {
            try {
                MatrixXd img = loadImage(image_paths[i], target_height, target_width);
                images.push_back(img);
                
                if (!image_labels.empty() && i < image_labels.size()) {
                    labels.push_back(image_labels[i]);
                } else {
                    labels.push_back("image_" + to_string(i));
                }
                
            } catch (const exception& e) {
                cerr << "Error loading " << image_paths[i] << ": " << e.what() << endl;
            }
        }
        
        cout << "Loaded " << images.size() << " images" << endl;
    }
    
    // Convertir toutes les images en une grande matrice (flatten)
    MatrixXd flattenImages() const {
        int num_images = images.size();
        int features = image_height * image_width;
        
        MatrixXd dataset(num_images, features);
        
        for (int i = 0; i < num_images; ++i) {
            // Aplatir l'image en vecteur ligne
            Map<const RowVectorXd> flattened(images[i].data(), features);
            dataset.row(i) = flattened;
        }
        
        return dataset;
    }
    
    // Afficher les statistiques du dataset
    void printStats() const {
        if (images.empty()) {
            cout << "No images loaded" << endl;
            return;
        }
        
        MatrixXd flat_data = flattenImages();
        
        cout << "Dataset Statistics:" << endl;
        cout << "Number of images: " << images.size() << endl;
        cout << "Image dimensions: " << image_height << "x" << image_width << endl;
        cout << "Features per image: " << image_height * image_width << endl;
        cout << "Pixel value range: [0, 1]" << endl;
        cout << "Mean pixel value: " << flat_data.mean() << endl;
        cout << "Std pixel value: " << sqrt((flat_data.array() - flat_data.mean()).square().sum() / (flat_data.size() - 1)) << endl;
    }
    
    // Getters
    const vector<MatrixXd>& getImages() const { return images; }
    const vector<string>& getLabels() const { return labels; }
    int getImageHeight() const { return image_height; }
    int getImageWidth() const { return image_width; }
};

// Fonctions supplémentaires pour la manipulation d'images
class ImageUtils {
public:
    // Normaliser les images
    static void normalizeDataset(vector<MatrixXd>& images) {
        if (images.empty()) return;
        
        // Calculer mean et std sur tout le dataset
        double total_mean = 0.0;
        double total_std = 0.0;
        int total_pixels = 0;
        
        for (const auto& img : images) {
            total_mean += img.sum();
            total_pixels += img.size();
        }
        total_mean /= total_pixels;
        
        for (const auto& img : images) {
            total_std += (img.array() - total_mean).square().sum();
        }
        total_std = sqrt(total_std / total_pixels);
        
        // Appliquer normalisation
        for (auto& img : images) {
            img = (img.array() - total_mean) / total_std;
        }
    }
    
    // Augmentation de données : miroir horizontal
    static MatrixXd horizontalFlip(const MatrixXd& image) {
        return image.rowwise().reverse();
    }
    
    // Augmentation de données : rotation 90°
    static MatrixXd rotate90(const MatrixXd& image) {
        return image.transpose().rowwise().reverse();
    }
    
    // Découper une partie de l'image
    static MatrixXd cropImage(const MatrixXd& image, int start_row, int start_col, int height, int width) {
        return image.block(start_row, start_col, height, width);
    }
};

std::vector<std::string> getJpegFiles(const std::string& directoryPath) {
    std::vector<std::string> jpegFiles;
    try {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            std::string extension = entry.path().extension().string();
            // Vérifier les extensions JPEG
            if (extension == ".jpg" || extension == ".jpeg" || 
                extension == ".JPG" || extension == ".JPEG") {
                jpegFiles.push_back(entry.path().string());
            }
        }
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Erreur: " << ex.what() << std::endl;
    }
    return jpegFiles;
}

vector<int> ordinalEncoding(vector<string>& classes, vector<string>& data_labels){
    vector<int> encoded_lab;

    for(string& lab : data_labels){
        // Parcourir les classes pour trouver l'index correspondant
        for(int i = 0; i < classes.size(); i++){
            if(lab == classes[i]){
                encoded_lab.push_back(i);  // Ajouter l'index de la classe
                break;  // Sortir de la boucle une fois trouvé
            }
        }
    }

    return encoded_lab;
}

template<typename T1, typename T2>
void shuffle_two_vectors(T1& vec1, T2& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    
    std::vector<size_t> indices(vec1.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    T1 temp_vec1(vec1.size());
    T2 temp_vec2(vec2.size());
    
    for (size_t i = 0; i < indices.size(); ++i) {
        temp_vec1[i] = vec1[indices[i]];
        temp_vec2[i] = vec2[indices[i]];
    }
    
    vec1 = std::move(temp_vec1);
    vec2 = std::move(temp_vec2);
}




// Exemple d'utilisation
ImageDataset loadDataSet() {
    ImageDatasetLoader loader;
    vector<string> image_labels;
    vector<string> image_paths;
  
    for(String path : class_path) {
        auto class_image = getJpegFiles(BASE_DATA_PATH + path);
        for(size_t i = 0; i < class_image.size(); i++){
            image_labels.push_back(path);
        }
        image_paths.insert(image_paths.end(), class_image.begin(), class_image.end());
    }
    
    try { 
        // Charger le dataset
        loader.loadDataset(image_paths, image_labels, 128, 128);
        loader.printStats();
        
        // Obtenir les données sous forme de matrice
        MatrixXd dataset = loader.flattenImages();
        cout << "Dataset shape: " << dataset.rows() << " x " << dataset.cols() << endl;
        
        vector<MatrixXd> X = loader.getImages();
        vector<int> Y = ordinalEncoding(class_path, image_labels);
        shuffle_two_vectors(X, Y);

        

        // Afficher la première image
        if (!loader.getImages().empty()) {
            cout << "First image matrix (10x10 block):\n" 
                 << "Label : " + class_path[Y[0]] + "\n"
                 << X[0].block(0, 0, 10, 10) << endl;
        }
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    
    return ImageDataset(class_path, loader.getImages(), image_labels);
}