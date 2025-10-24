# Guide Complet de Débogage avec GDB

## Installation de GDB

```bash
# Sur Ubuntu/Debian
sudo apt install gdb

# Sur CentOS/RHEL/Fedora
sudo yum install gdb
# ou
sudo dnf install gdb

# Sur macOS (avec Homebrew)
brew install gdb
# Note: sur macOS,可能需要额外配置 https://sourceware.org/gdb/wiki/PermissionsDarwin
```

### Clear the command line
Ctrl + L / shell clear

## 1. Compiler avec les informations de débogage

### Avec CMake
```cmake
# Dans votre CMakeLists.txt
set(CMAKE_BUILD_TYPE Debug)  # Pour avoir les symboles de debug

# Ou depuis la ligne de commande :
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Avec g++ directement
```bash
g++ -g -o my_program my_program.cpp
```

## 2. Démarrer GDB et charger votre programme

```bash
# Lancer GDB avec votre exécutable
gdb ./src/CNN
# ou
gdb ./test/CNNtest
# ou
gdb my_program

# Interface graphique TUI
gdb -tui ./src/CNN
```

## 3. Définir des points d'arrêt (Breakpoints)

```gdb
break main                    # Point d'arrêt sur la fonction main()
break convolution.cpp:45      # Point d'arrêt à la ligne 45
break ConvLayer::forward      # Point d'arrêt sur une méthode
break DenseLayer::backward    # Point d'arrêt sur une méthode spécifique

# Points d'arrêt conditionnels
break convolution.cpp:100 if input_size == 0
break main if argc > 1

# Points d'arrêt temporaires (se suppriment après utilisation)
tbreak function_name

# Gestion des points d'arrêt
info breakpoints              # Liste tous les points d'arrêt
delete 2                      # Supprime le point d'arrêt numéro 2
delete                        # Supprime tous les points d'arrêt
disable 1                     # Désactive le point d'arrêt 1
enable 1                      # Réactive le point d'arrêt 1
```

## 4. Exécuter votre programme

```gdb
run                           # Exécute le programme
run arg1 arg2                 # Exécute avec des arguments
run --dataset data/ --epochs 10  # Avec des arguments longs

# Avec GDB déjà lancé
gdb --args ./src/CNN --dataset data/ --epochs 10
(gdb) run
```

## 5. Contrôler le flux d'exécution

```gdb
continue                      # Continue l'exécution jusqu'au prochain point d'arrêt
next                          # Pas à pas (saute les fonctions)
step                          # Pas à pas (entre dans les fonctions)
finish                        # Termine la fonction actuelle
until line_number             # Continue jusqu'à la ligne spécifiée
until                         # Continue jusqu'à la ligne suivante

# Exécution pas à pas
stepi                         # Pas à pas instruction assembleur
nexti                         # Pas à pas instruction (saute les appels)
```

## 6. Inspecter votre programme

### Afficher variables et données
```gdb
print variable_name           # Affiche une variable
print matrix.size()           # Affiche le résultat d'une fonction
print *pointer                # Déréférence un pointeur
print array[0]@10            # Affiche 10 éléments d'un tableau

# Pour Eigen (matrices)
print eigen_matrix.rows()
print eigen_matrix.cols()
print eigen_matrix(0,0)

# Afficher plusieurs variables
info locals                   # Affiche toutes les variables locales
info args                     # Affiche les arguments de la fonction

# Formats d'affichage
print/x variable              # Hexadecimal
print/d variable              # Décimal
print/t variable              # Binaire
print variable                # Format automatique
```

### Pile d'appels et code
```gdb
backtrace                     # Affiche la pile d'appels
where                         # Affiche où on est dans le code
frame 2                       # Change le cadre de la pile d'appels
info frame                    # Informations sur le cadre actuel

list                          # Affiche le code autour de la position actuelle
list 10,20                    # Affiche les lignes 10 à 20
list function_name            # Affiche le code d'une fonction
```

## 7. Commandes avancées

### Points de surveillance (Watchpoints)
```gdb
watch variable_name           # Arrête quand la variable change
watch *0x12345678            # Surveillance d'adresse mémoire
rwatch variable              # Arrête quand la variable est lue
awatch variable              # Arrête quand la variable est lue ou écrite

info watchpoints             # Liste les points de surveillance
```

### Modifier l'exécution
```gdb
set var variable_name = value # Modifie une variable
set var i = 0                # Réinitialise un compteur
set var *ptr = 100           # Modifie via un pointeur

jump line_number             # Saute à une ligne spécifique
return                       Force le retour de la fonction actuelle
return value                 # Retourne avec une valeur
```

### Debug multi-thread
```gdb
info threads                 # Liste tous les threads
thread 2                     # Passe au thread 2
thread apply all command     # Exécute une commande sur tous les threads
break line thread 2          # Point d'arrêt spécifique à un thread
```

## 8. Exemple de session de débogage complète

```bash
# Compilation
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Débogage
gdb ./src/CNN

# Dans GDB :
(gdb) break main
(gdb) break ConvLayer::forward
(gdb) run --dataset data/ --epochs 5
(gdb) next
(gdb) print input_matrix
(gdb) continue
(gdb) step
(gdb) print filters[0]
(gdb) backtrace
(gdb) watch loss_value
(gdb) continue
(gdb) quit
```

## 9. Débogage des tests

```bash
gdb ./test/CNNtest

# Points d'arrêt pour les tests Google Test :
(gdb) break ConvLayerTest_ConstructorInitializesCorrectly_Test::TestBody
(gdb) break DenseLayerTest_ForwardPass_Test::TestBody
(gdb) run
```

## 10. Scripts de débogage automatique

Créez un fichier `debug_commands.gdb` :
```gdb
# debug_commands.gdb
set pagination off
break main
run
break ConvLayer::forward
continue
print input
step
print output
continue
```

Exécutez le script :
```bash
gdb -x debug_commands.gdb ./src/CNN
```

## 11. Commandes utiles pour le débogage de CNN

```gdb
# Points d'arrêt typiques pour un CNN
break main
break ConvLayer::ConvLayer
break ConvLayer::forward
break ConvLayer::backward
break DenseLayer::forward
break PoolLayer::forward
break ImgDataset::loadImage

# Surveillance de variables importantes
watch loss_value
watch accuracy
watch gradients[0]
```

## 12. Gestion des signaux et exceptions

```gdb
handle SIGSEGV stop print     # Arrête sur les segfaults
handle SIGTERM nostop         # Ignore SIGTERM
catch throw                   # Arrête sur les exceptions C++
catch catch                   # Arrête sur la capture d'exceptions
```

## 13. Tips et bonnes pratiques

1. **Toujours compiler avec `-g`** pour avoir les symboles de débogage
2. **Utiliser `step` pour entrer dans les fonctions**, `next` pour les sauter
3. **`backtrace` est essentiel** pour comprendre les crashes
4. **Les watchpoints** sont utiles pour trouver où une variable est modifiée
5. **Utiliser `layout src`** dans GDB TUI pour une interface plus visuelle
6. **Sauvegarder l'historique** avec `set history save on`

## 14. Commandes de sortie et journalisation

```gdb
set logging on                # Active la journalisation
set logging file gdb.log      # Fichier de log
set logging overwrite on      # Écrase le fichier existant

echo Message de debug\n       # Affiche un message
printf "Valeur: %d\n", variable  # Formatage avancé
```

Avec ce guide complet, vous disposez de toutes les commandes nécessaires pour déboguer efficacement votre CNN et comprendre exactement ce qui se passe dans chaque couche !