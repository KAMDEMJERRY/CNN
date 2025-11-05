#ifndef IMGDATASET_HPP
#define IMGDATASET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp> // <-- Then include OpenCV's eigen header
#include <opencv2/core.hpp> 
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <random>
#include <utility>
#include <numeric> // for std::iota

using namespace Eigen;
using namespace std;
using namespace cv;
namespace fs = std::filesystem;


// Variables globales externes
extern String BASE_DATA_PATH;
extern vector<String> class_path;



// Déclaration de la classe ImageDatasetLoader
class ImageDatasetLoader {
private:
    string dataset_path;
    vector<vector<MatrixXd>> images;
    vector<string> classes;// Déclaration des fonctions externes
    vector<string> labels;// Déclaration des fonctions externes
    string color_mode;
    int image_height;
    int image_width;

public:

    // Constructeur
    ImageDatasetLoader(string dataset_path, int img_size ,string color_mode = "RGB");
    ImageDatasetLoader()= default;
 
    // Méthodes
    vector<MatrixXd> loadImage(string image_path,  int target_height = -1, int target_width = -1);

    void loadDataset(
                     int target_height = 28, 
                     int target_width = 28);
    
    // Getters
    const std::vector<std::vector<MatrixXd>>& getImages() const;
    const std::vector<string>& getLabels() const;
    const std::vector<string>& getClasses() const;

    int getImageHeight() const;
    int getImageWidth() const;
    void afficherImageEigenNormalisee(const Eigen::MatrixXd &r, const Eigen::MatrixXd &g, const Eigen::MatrixXd &b);
};



// Déclaration de la classe ImageDataset
class ImageDataset {
public:
    vector<string> classes;
    vector<vector<MatrixXd>> images;
    vector<string> labels;
    VectorXi encoded_labels;
    float split = 0.8; // Pourcentage de données pour l'entraînement
    int channels ; // Nombre de canaux (RGB)
    int image_size; // Taille des images (assumées carrées)
    ImageDatasetLoader loader;
    
    




    // Constructeur
    // ImageDataset( 
    //     vector<String> classes,
    //     vector<MatrixXd> images,
    //     vector<String> labels
    // );
    ImageDataset(
        string dataset_path,
        int img_size,
        string color_mode = "GARY"
    );
    std::vector<std::vector<MatrixXd>> normalize();
    ImageDataset() = default;

    // Méthodes
    VectorXi ordinalEncoding(vector<string>& classes, vector<string>& data_labels);
    void shuffle_dataset();
    
    // Getters
    vector<vector<MatrixXd>> getX();
    std::pair<vector<string>, VectorXi> getY();
    pair<vector<vector<MatrixXd>>, VectorXi> getTrain();
    pair<vector<vector<MatrixXd>>, VectorXi> getTest();

    void summary();
};



// Déclaration de la classe ImageUtils
class ImageUtils {
public:
    // Méthodes statiques
    static void normalizeDataset(vector<MatrixXd>& images);
    static MatrixXd horizontalFlip(const MatrixXd& image);
    static MatrixXd rotate90(const MatrixXd& image);
    static MatrixXd cropImage(const MatrixXd& image, int start_row, int start_col, int height, int width);
};

// Déclarations des fonctions
std::vector<std::string> getJpegFiles(const std::string& directoryPath);
std::vector<std::string> getDirectoriesInDirectory(const fs::path &directoryPath);
vector<int> ordinalEncoding(vector<string> &classes, vector<string> &data_labels);


// Add these overloaded reorder functions
void reorder(std::vector<std::vector<Eigen::MatrixXd>> &vec, const std::vector<size_t> &indices);
void reorder(std::vector<std::string> &vec, const std::vector<size_t> &indices);
void reorder(Eigen::VectorXi &vec, const std::vector<size_t> &indices);

// Déclaration du template
template<typename T1, typename T2>
void shuffle_two_vectors(T1& vec1, T2& vec2);

// Déclaration de la fonction principale
ImageDataset loadDataSet();

#endif // IMGDATASET_HPP