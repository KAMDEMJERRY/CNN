#!/bin/bash


clear

## compile 
cd ..
# cmake --build build --clean-first
cmake --build build
ls
cd build/test/


# run unit test
cd ../build/test/
ctest --output-on-failure  # Affiche la sortie seulement si échec
cd ../../script
#.CNNtest

# or with ctest

# ctest                   # Depuis le répertoire build/
# ctest                   # Exécute tous les tests
# ctest -V                # Mode verbose (affiche tous les détails)
# ctest --output-on-failure  # Affiche la sortie seulement si échec
# ctest -R CNN_Tests      # Exécute un test spécifique par nom
