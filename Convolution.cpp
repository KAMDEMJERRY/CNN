#include <iostream>
#include <fstream>
#include <Eigen/Dense>
using namespace Eigen;



int img[5][5] = {
    {3, 0, 1, 2, 7},
    {1, 5, 8, 9, 3},
    {2, 7, 2, 5, 1},
    {0, 1, 3, 1, 7},
    {4, 2, 1, 6, 2}
};
int filter[3][3] = {
    {1, 0, -1},
    {1, 0, -1},
    {1, 0, -1}
};
int padding = 1;
int stride = 1;
int output[5][5] = {0};
int pooled[3][3] = {0};
int flat[9] = {0};
int convolution() {
    // Convolution implementation goes here
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            for(int m = 0; m < 3; m++) {
                for(int n = 0; n < 3; n++) {
                    int x = i + m - padding;
                    int y = j + n - padding;
                    if(x >= 0 && x < 5 && y >= 0 && y < 5) {
                        output[i][j] += img[x][y] * filter[m][n];
                    }
                }
            }
        }
    }
    return 0;
}
void polling() {
    // Polling implementation goes here
    for(int i = 0; i < 5; i += 2) {
        for(int j = 0; j < 5; j += 2) {
            int maxVal = output[i][j];
            for(int m = 0; m < 2; m++) {
                for(int n = 0; n < 2; n++) {
                    if(i + m < 5 && j + n < 5) {
                        if(output[i + m][j + n] > maxVal) {
                            maxVal = output[i + m][j + n];
                        }
                    }
                }
            }
            pooled[i / 2][j / 2] = maxVal;
        }
    }
}
void flatten() {
    // Flattening implementation goes here
    int index = 0;
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            flat[index++] = pooled[i][j];
        }
    }
}
void ecrireFichier() {
    std::ofstream file("conv_output.txt");
    if(file.is_open()) {
        for(int i = 0; i < 9; i++) {
            file << flat[i] << ", "; 
        }
        file << std::endl;
        file.close();
    }
}
void lireFichier() {
    std::ifstream file("conv_output.txt");
    if(file.is_open()) {
        for(int i = 0; i < 9; i++) {
            file >> flat[i];
        }
        file.close();
    }
}

void printOutput() {
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            std::cout << output[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
void printPooled() {
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            std::cout << pooled[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
void printFlat() {
    for(int i = 0; i < 9; i++) {
        std::cout << flat[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Convolution Output:" << std::endl;
    convolution();
    printOutput();

    std::cout << "Pooled Output:" << std::endl;
    polling();
    printPooled();

    std::cout << "Flattened Output:" << std::endl;
    flatten();
    printFlat();
    

    ecrireFichier();
    lireFichier();
    std::cout << "Flattened Output after reading from file:" << std::endl;
    printFlat();

    return 0;
}

