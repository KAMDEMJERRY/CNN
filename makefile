# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -g -I/usr/include/eigen3

# Fichiers source et objets
SRCS = main.cpp convolution.cpp imgdataset.cpp dense.cpp utils.cpp
OBJS = $(addprefix ./build/, $(SRCS:.cpp=.o))
TARGET = ./build/CNN.exe

# Règle par défaut
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) `pkg-config --cflags --libs opencv4` -lstdc++fs

# Règle pour les fichiers objets
./build/%.o: %.cpp convolution.hpp imgdataset.hpp
	@mkdir -p ./build
	$(CXX) $(CXXFLAGS) -c $< -o $@ `pkg-config --cflags opencv4`


# Nettoyage
clean:
	rm -rf ./build

# Réinstallation complète
re: clean $(TARGET)

.PHONY: clean re

# MLP: MLP.cpp
# 	g++ -o ./build/MLP MLP.cpp -I /usr/include/eigen3

# test: test.cpp
# 	g++ -o ./build/test test.cpp -I /usr/include/eigen3

# clean:
# 	rm -f /build/*

# g++ -std=c++11 imgdataset.cpp -o ./build/imgdataset  -I /usr/include/eigen3 `pkg-config --cflags --libs opencv4` -lstdc++fs