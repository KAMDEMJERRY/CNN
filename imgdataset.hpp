#ifndef IMGDATASET_HPP
#define IMGDATASET_HPP

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

// Variables globales externes
extern String BASE_DATA_PATH;
extern vector<String> class_path;

// Déclaration de la classe ImageDataset
class ImageDataset {
public:
    vector<String> classes;
    vector<MatrixXd> images;
    vector<String> labels;
    vector<int> encoded_labels;

    // Constructeur
    ImageDataset( 
        vector<String> classes,
        vector<MatrixXd> images,
        vector<String> labels
    );

    // Méthodes
    vector<int> ordinalEncoding(vector<string>& classes, vector<string>& data_labels);
    vector<MatrixXd> getX();
    vector<string> getY();
    vector<int> getY_encoded();
};

// Déclaration de la classe ImageDatasetLoader
class ImageDatasetLoader {
private:
    vector<MatrixXd> images;
    vector<string> labels;// Déclaration des fonctions externes
    int image_height;
    int image_width;

public:
    // Méthodes
    MatrixXd loadImage(const string& image_path, int target_height = -1, int target_width = -1);
    void loadDataset(const vector<string>& image_paths, 
                    const vector<string>& image_labels = {},
                    int target_height = 64, 
                    int target_width = 64);
    MatrixXd flattenImages() const;
    void printStats() const;
    
    // Getters
    const vector<MatrixXd>& getImages() const;
    const vector<string>& getLabels() const;
    int getImageHeight() const;
    int getImageWidth() const;
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
vector<int> ordinalEncoding(vector<string>& classes, vector<string>& data_labels);

// Déclaration du template
template<typename T1, typename T2>
void shuffle_two_vectors(T1& vec1, T2& vec2);

// Déclaration de la fonction principale
ImageDataset loadDataSet();

#endif // IMGDATASET_HPP