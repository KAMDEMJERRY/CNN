## La parallelisation du CNN



1. Parallelisation des donnees

Pour la partie convolutive il suffira de repartir les images en lots
et les regrouper a la fin pour constituer la matrice finale qui sera passe en entree a la couche dense. Dans un second temps on vera dans quelle mesure diviser les matrices representant les inputs pour paralleliser les operation de :

- convolution
- pooling
- activation

Pour la partie dense on va separer les donnees en batchs et faire du batch gradient descents a l'entrainement puis faire une moyenne des loss obtenues
Dans un second temps on vera comment activer les option de parallelisation des produits matricielles qui sont des operation vectorielles avec les options proposees par la librairies EIGEN


1. Parallelisation des traitements

La sequence des couche ne peux pas subir de parallelisation par ce les caracteristique d'une couche dans une architecture sequentielle est qu'elle depend de la couche precedente. Donc tant que la couche precedente n'a pas fini sont traitement elle ne peux rien effectuer comme traitement parallele.
La parallelisation a ce niveau de fera sur les fonction internes tels que la convolution

`Peux t'on paralleliser l'operation de convolution de pooling, d'activation`


Pour la partie dense on vera comment activer les options de parallelisation des traitements des produits matricielles qui sont des operation vectorielles avec les options proposees par la librairies EIGEN




## Methodologies d'evaluation de la parallelisation

En partant des algorithme les plus atomiques parallelisables
On va realiser des methodes qui creer des donnees fictives et evaluer le temps d'execution puis loguer le pour chaque fonction le resultats et les parametres de test(num threads utilisees) dans les un fichier predefinit pour la methode evalue.

Par la suite on evaluera le temps entier d'entrainement avec et sans parallelisation et le temps.

et le temps du prediction pour une image.




































