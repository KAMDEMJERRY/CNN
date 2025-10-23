clear

## compile 
make --build build
# make --build build --clear-first

# run unit test
cd ../build/test/
#.CNNtest

# or with ctest

# ctest                   # Depuis le répertoire build/
# ctest                   # Exécute tous les tests
# ctest -V                # Mode verbose (affiche tous les détails)
# ctest --output-on-failure  # Affiche la sortie seulement si échec
# ctest -R CNN_Tests      # Exécute un test spécifique par nom
