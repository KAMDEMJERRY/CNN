# CNN
Parallelisation d'un CNN

Ce depot retrace mes travaux sur la parallelisation d'un CNN pour la detection d'images medicales
- step 1 : CNN sequentiel 


## System
linux                                       22.04


## Tools

- g++                                        11.4.0
- libeigen3-dev                              3.4.0-2ubuntu2 

============
Finir le CNN

Augmenter le jeu de donnee

Paralleliser

Les performances metriques temporelle

Pose des questions ne reste pas dans ton coin
============

Voici d'excellentes ressources PDF pour débuter en parallélisation :

## 📚 Ressources Fondamentales

### 1. **Introduction Générale**
- **"Introduction au Parallel Programming"** - Blaise Barney (Lawrence Livermore)
  *Concepts de base, architectures, modèles de programmation*

### 2. **OpenMP (Parallélisme partagé)**
- **"OpenMP Tutorial"** - Tim Mattson (Intel)
  *Très pédagogique, nombreux exemples*
- **"OpenMP Application Programming Interface"** - OpenMP Architecture Review Board
  *Référence complète*

### 3. **MPI (Parallélisme distribué)**
- **"MPI Tutorial"** - Wesley Kendall (University of Tennessee)
  *Guide progressif*
- **"Using MPI"** - William Gropp
  *Classique du domaine*

## 🎯 Pour le Machine Learning

### 4. **Parallélisation en ML**
- **"Parallel Machine Learning with Eigen"** - Documentation Eigen
- **"Deep Learning Systems: Parallelism"** - CMU Lectures
  *Optimisation des réseaux de neurones*

### 5. **GPU Computing**
- **"CUDA C++ Programming Guide"** - NVIDIA
  *Pour l'accélération GPU*
- **"OpenCL Programming Guide"** - Aaftab Munshi
  *Alternative multi-plateforme*

## 🔍 Où les trouver :

1. **Site de Lawrence Livermore** : `hpc.llnl.gov/documentation/tutorials`
2. **OpenMP** : `openmp.org/wp-content/uploads/OpenMP-4.0-C.pdf`
3. **MPI** : `mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf`
4. **NVIDIA** : `docs.nvidia.com/cuda/cuda-c-programming-guide`

## 💡 Conseils de Démarrage :

1. **Commence par OpenMP** (plus simple)
2. **Pratique sur ton code Eigen existant**
3. **Mesure les performances** avant/après
4. **Debug avec des petits jeux de données**

**Veux-tu que je te montre comment paralléliser des parties spécifiques de ton code CNN ?** 🚀

*PS: Je peux aussi te donner des extraits de code pour paralléliser tes opérations matricielles Eigen !*