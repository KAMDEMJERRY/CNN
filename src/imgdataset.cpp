#include "imgdataset.hpp"

// Définition des variables globales
String BASE_DATA_PATH = "../../../dataset/bloodcellsub/images/TRAIN/";
vector<String> class_path = {"EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"};



ImageDataset::ImageDataset(string dataset_path) : loader(dataset_path){
    images = loader.getImages();
    labels = loader.getLabels();
    classes = loader.getClasses();
    ordinalEncoding(classes, labels);
    shuffle_dataset();
}

VectorXi ImageDataset::ordinalEncoding(vector<string>& classes, vector<string>& data_labels){
    encoded_labels.resize(data_labels.size());
    int j = 0;
    for(string& lab : data_labels){
        for(int i = 0; i < classes.size(); i++){
                // std::cout << "Encoding label: " << lab << " as " << i << std::endl;
                // std::cout << "Comparing with class: " << classes[i] << std::endl;
                // std::cout << "Match result: " << (lab == classes[i] ? "MATCH" : "NO MATCH") << std::endl;
            if(lab == classes[i]){
                encoded_labels(j++) = i;
                break;
            }
        }
    }

    return encoded_labels;
}

// vector<MatrixXd> ImageDataset::getX(){
//     return images;
// }

// vector<string> ImageDataset::getY(){
//     return labels;
// }

// vector<int> ImageDataset::getY_encoded(){
//     return encoded_labels;
// }



ImageDatasetLoader::ImageDatasetLoader(string dataset_path):dataset_path(dataset_path)
{
    loadDataset(dataset_path, 128, 128);
}

// Implémentation de ImageDatasetLoader
vector<MatrixXd> ImageDatasetLoader::loadImage(const string& image_path, int target_height, int target_width) {
    assert(target_height > 0 && target_width > 0 && "Target dimensions must be positive");
   
    Mat image = imread(image_path, IMREAD_COLOR);
    // std::cout << "Loading image: " << image_path << " ("
    //           << image.cols << "x" << image.rows << ")" << std::endl;

    assert(image.empty() == false && "Image loading failed");

    std::vector<cv::Mat> canaux;
    
    if (target_height > 0 && target_width > 0) {
        Mat resized_image;
        resize(image, resized_image, Size(target_width, target_height));
        image = resized_image;
    }

    cv::split(image, canaux);

    
    MatrixXd RED_image(image.rows, image.cols);
    MatrixXd GREEN_image(image.rows, image.cols);
    MatrixXd BLUE_image(image.rows, image.cols);
    vector<MatrixXd> eigen_images;
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            RED_image(i, j) = static_cast<double>(canaux[2].at<uchar>(i, j)) / 255.0;
            GREEN_image(i, j) = static_cast<double>(canaux[1].at<uchar>(i, j)) / 255.0;
            BLUE_image(i, j) = static_cast<double>(canaux[0].at<uchar>(i, j)) / 255.0;
        }
    }
    eigen_images.push_back(RED_image);
    eigen_images.push_back(GREEN_image);
    eigen_images.push_back(BLUE_image);
    
    // std::cout << "Image loaded and converted to Eigen matrices." << std::endl;
    // std::cout << "Dimensions - R: " << RED_image.rows() << "x" << RED_image.cols()
    //           << ", G: " << GREEN_image.rows() << "x" << GREEN_image.cols()
    //           << ", B: " << BLUE_image.rows() << "x" << BLUE_image.cols() << std::endl;
    // std::cout << "----------------------------------------" << std::endl;
    // std::cout << eigen_images.size() << " channels loaded." << std::endl;
    
    return eigen_images;
}


void ImageDatasetLoader::loadDataset(const string &dataset_path, int target_height, int target_width)
{
    this->dataset_path = dataset_path;
    images.clear();
    labels.clear();
    image_height = target_height;
    image_width = target_width;
    std::cout << "Loading dataset from: " << dataset_path << std::endl;
    std::vector<std::string> classes_path = getDirectoriesInDirectory(fs::path(dataset_path));
    std::cout << std::endl;


    classes.clear();
    for (const auto &cls : classes_path) {
        // Prendre la derniere partie du chemin comme nom de classe
        string classe = cls.substr(cls.find_last_of("/\\") + 1);
        classes.push_back(classe);
        fs::path class_dir = fs::path(dataset_path) / classe;
        // std::cout << "Loading images for class: " << classe <<  " at dir path: " << class_dir << std::endl;

        vector<string> class_image_paths = getJpegFiles(class_dir.string());
        
        for (const auto &img_path : class_image_paths) {
            try {
                std::vector<MatrixXd> img = loadImage(img_path, target_height, target_width);
                images.push_back(img);
                labels.push_back(classe);
            } catch (const exception &e) {
                cerr << "Error loading " << img_path << ": " << e.what() << endl;
            }
        }
    }
    // std::cout << "Dataset loaded. Total images: " << images.size() << std::endl;
}





const vector<vector<MatrixXd>>& ImageDatasetLoader::getImages() const { return images; }
const vector<string>& ImageDatasetLoader::getLabels() const { return labels; }
const std::vector<string> &ImageDatasetLoader::getClasses() const
{
    return this->classes;
}
int ImageDatasetLoader::getImageHeight() const { return image_height; }
int ImageDatasetLoader::getImageWidth() const { return image_width; }

// Implémentation de ImageUtils
void ImageUtils::normalizeDataset(vector<MatrixXd>& images) {
    if (images.empty()) return;
    
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
    
    for (auto& img : images) {
        img = (img.array() - total_mean) / total_std;
    }
}

MatrixXd ImageUtils::horizontalFlip(const MatrixXd& image) {
    return image.rowwise().reverse();
}

MatrixXd ImageUtils::rotate90(const MatrixXd& image) {
    return image.transpose().rowwise().reverse();
}

MatrixXd ImageUtils::cropImage(const MatrixXd& image, int start_row, int start_col, int height, int width) {
    return image.block(start_row, start_col, height, width);
}

// Implémentation des fonctions
std::vector<std::string> getJpegFiles(const std::string& directoryPath) {
    std::vector<std::string> jpegFiles;
    try {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            std::string extension = entry.path().extension().string();
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

std::vector<std::string> getDirectoriesInDirectory(const fs::path& directoryPath) {
    std::vector<std::string> filePaths;
    try {
        for (const auto& entry : fs::directory_iterator(directoryPath)) {
            if (entry.is_directory()) {
                filePaths.push_back(entry.path());
            }
        }
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "Erreur: " << ex.what() << std::endl;
    }
    return filePaths;
}

// 

void ImageDataset::shuffle_dataset()
{
      // 1. Créer un vecteur d'indices (0, 1, 2, ...)
        std::vector<size_t> indices(images.size());
        std::iota(indices.begin(), indices.end(), 0);

        // 2. Mélanger le vecteur d'indices avec un générateur de nombres aléatoires de qualité
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // 3. Réorganiser les vecteurs originaux en utilisant les indices mélangés
        reorder(images, indices);
        reorder(labels, indices);
        reorder(encoded_labels, indices);
}

vector<vector<MatrixXd>> ImageDataset::getX()
{
    return this->images;
}

std::pair<vector<string>, VectorXi> ImageDataset::getY()
{
    return std::pair<vector<string>, VectorXi>(this->labels, this->encoded_labels);
}

std::pair<std::vector<std::vector<Eigen::MatrixXd>>, Eigen::VectorXi> ImageDataset::getTrain() {
    size_t train_size = static_cast<size_t>(this->split * this->images.size());
    
    std::vector<std::vector<Eigen::MatrixXd>> train_images(
        this->images.begin(), 
        this->images.begin() + train_size
    );
    
    Eigen::VectorXi train_labels = this->encoded_labels.head(train_size);
    
    return std::make_pair(train_images, train_labels);
}

std::pair<std::vector<std::vector<Eigen::MatrixXd>>, Eigen::VectorXi> ImageDataset::getTest() {
    size_t train_size = static_cast<size_t>(this->split * this->images.size());
    
    std::vector<std::vector<Eigen::MatrixXd>> test_images(
        this->images.begin() + train_size, 
        this->images.end()
    );
    
    Eigen::VectorXi test_labels = this->encoded_labels.tail(this->images.size() - train_size);
    
    return std::make_pair(test_images, test_labels);
}

void ImageDataset::summary()
{
    cout << "Dataset Summary:" << endl;
    cout << "Number of classes: " << classes.size() << endl;
    cout << "Number of images: " << images.size() << endl;
    cout << "Image dimensions: " << loader.getImageHeight() << "x" << loader.getImageWidth() << endl;
      
    cout << "Classes: ";
    for (const auto& cls : this->classes) {
        cout << cls << " ";
    }
    cout << "\n\n";

            // Prendre la première image comme exemple
    vector<MatrixXd> first_image = images[0];
    cout << "Première image (extrait 10x10 du canal rouge):\n" << first_image[0].block(0, 0, 10, 10) << "\n\n";
    cout << "Première image (extrait 10x10 du canal vert):\n" << first_image[1].block(0, 0, 10, 10) << "\n\n";
    cout << "Première image (extrait 10x10 du canal bleu):\n" << first_image[2].block(0, 0, 10, 10) << "\n\n";

    this->loader.afficherImageEigenNormalisee(first_image[0], first_image[1], first_image[2]);

    cout << "Label de la première image: " << labels[0] << "\n\n";

}

void reorder(std::vector<std::vector<Eigen::MatrixXd>> &vec, const std::vector<size_t> &indices) {
    std::vector<std::vector<Eigen::MatrixXd>> new_vec(vec.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_vec[i] = vec[indices[i]];
    }
    vec = new_vec;
}

void reorder(std::vector<std::string> &vec, const std::vector<size_t> &indices) {
    std::vector<std::string> new_vec(vec.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_vec[i] = vec[indices[i]];
    }
    vec = new_vec;
}

void reorder(Eigen::VectorXi &vec, const std::vector<size_t> &indices) {
    Eigen::VectorXi new_vec(vec.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        new_vec[i] = vec[indices[i]];
    }
    vec = new_vec;
}





void ImageDatasetLoader::afficherImageEigenNormalisee(const Eigen::MatrixXd& r, const Eigen::MatrixXd& g, const Eigen::MatrixXd& b) {
    try {
        // Vérifier les dimensions
        if (r.rows() != g.rows() || r.rows() != b.rows() || 
            r.cols() != g.cols() || r.cols() != b.cols()) {
            std::cerr << "Erreur: Les canaux RGB ont des dimensions différentes" << std::endl;
            return;
        }
        
        // Vérifier si les matrices sont vides
        if (r.rows() == 0 || r.cols() == 0) {
            std::cerr << "Erreur: Les matrices sont vides" << std::endl;
            return;
        }
        
        int height = r.rows();
        int width = r.cols();
        
        std::cout << "Dimensions de l'image: " << height << "x" << width << std::endl;
        std::cout << "Plage des valeurs - R: [" << r.minCoeff() << ", " << r.maxCoeff() << "]" 
                  << " G: [" << g.minCoeff() << ", " << g.maxCoeff() << "]"
                  << " B: [" << b.minCoeff() << ", " << b.maxCoeff() << "]" << std::endl;
        
        // Convertir les matrices Eigen en Mat OpenCV
        cv::Mat r_mat, g_mat, b_mat;
        cv::eigen2cv(r, r_mat);
        cv::eigen2cv(g, g_mat);
        cv::eigen2cv(b, b_mat);
        
        // Normaliser vers 0-255
        cv::Mat r_norm, g_norm, b_norm;
        
        // Si les valeurs sont déjà entre 0 et 1
        if (r.maxCoeff() <= 1.0 && g.maxCoeff() <= 1.0 && b.maxCoeff() <= 1.0) {
            r_mat.convertTo(r_norm, CV_8UC1, 255.0);
            g_mat.convertTo(g_norm, CV_8UC1, 255.0);
            b_mat.convertTo(b_norm, CV_8UC1, 255.0);
        } else {
            // Normalisation automatique
            double min_val = std::min({r.minCoeff(), g.minCoeff(), b.minCoeff()});
            double max_val = std::max({r.maxCoeff(), g.maxCoeff(), b.maxCoeff()});
            
            if (max_val > min_val) {
                r_mat.convertTo(r_norm, CV_8UC1, 255.0/(max_val-min_val), -255.0*min_val/(max_val-min_val));
                g_mat.convertTo(g_norm, CV_8UC1, 255.0/(max_val-min_val), -255.0*min_val/(max_val-min_val));
                b_mat.convertTo(b_norm, CV_8UC1, 255.0/(max_val-min_val), -255.0*min_val/(max_val-min_val));
            } else {
                // Cas où toutes les valeurs sont identiques
                r_mat.convertTo(r_norm, CV_8UC1, 255.0);
                g_mat.convertTo(g_norm, CV_8UC1, 255.0);
                b_mat.convertTo(b_norm, CV_8UC1, 255.0);
            }
        }
        
        // Fusionner les canaux (BGR pour OpenCV)
        std::vector<cv::Mat> canaux = {b_norm, g_norm, r_norm};
        cv::Mat image_bgr;
        cv::merge(canaux, image_bgr);
        
        // Redimensionner si l'image est trop grande pour l'affichage
        cv::Mat image_display;
        if (width > 800 || height > 600) {
            double scale = std::min(800.0/width, 600.0/height);
            cv::resize(image_bgr, image_display, cv::Size(), scale, scale, cv::INTER_LINEAR);
            std::cout << "Image redimensionnée à: " << image_display.cols << "x" << image_display.rows << std::endl;
        } else {
            image_display = image_bgr;
        }
        
        // Afficher l'image
        cv::imshow("Image Eigen Normalisée", image_display);
        std::cout << "Appuyez sur une touche pour fermer la fenêtre..." << std::endl;
        cv::waitKey(0);
        cv::destroyAllWindows();
        
    } catch (const cv::Exception& e) {
        std::cerr << "Erreur OpenCV: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erreur: " << e.what() << std::endl;
    }
}
// // Implémentation du template
// template<typename T1, typename T2>
// void shuffle_two_vectors(T1& vec1, T2& vec2) {
//     if (vec1.size() != vec2.size()) {
//         throw std::invalid_argument("Vectors must have the same size");
//     }
    
//     std::vector<size_t> indices(vec1.size());
//     for (size_t i = 0; i < indices.size(); ++i) {
//         indices[i] = i;
//     }
    
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::shuffle(indices.begin(), indices.end(), gen);
    
//     T1 temp_vec1(vec1.size());
//     T2 temp_vec2(vec2.size());
    
//     for (size_t i = 0; i < indices.size(); ++i) {
//         temp_vec1[i] = vec1[indices[i]];
//         temp_vec2[i] = vec2[indices[i]];
//     }
    
//     vec1 = std::move(temp_vec1);
//     vec2 = std::move(temp_vec2);
// }

// // Instanciation explicite du template pour les types couramment utilisés
// template void shuffle_two_vectors<vector<MatrixXd>, vector<int>>(vector<MatrixXd>&, vector<int>&);
// template void shuffle_two_vectors<vector<MatrixXd>, vector<string>>(vector<MatrixXd>&, vector<string>&);
// template void shuffle_two_vectors<vector<int>, vector<int>>(vector<int>&, vector<int>&);
// template void shuffle_two_vectors<vector<string>, vector<string>>(vector<string>&, vector<string>&);

// // Implémentation de la fonction principale
// ImageDataset loadDataSet() {
    
//     ImageDatasetLoader loader;
//     vector<string> image_labels;
//     vector<string> image_paths;
  
//     for(String path : class_path) {
//         auto class_image = getJpegFiles(BASE_DATA_PATH + path);
//         for(size_t i = 0; i < class_image.size(); i++){
//             image_labels.push_back(path);
//         }
//         image_paths.insert(image_paths.end(), class_image.begin(), class_image.end());
//     }
    
//     try { 
//         loader.loadDataset(image_paths, image_labels, 128, 128);
//     } catch (const exception& e) {
//         cerr << "Error: " << e.what() << endl;
//     }
    
//     ImageDataset dataset(class_path, loader.getImages(), image_labels);
//     return dataset;
// }