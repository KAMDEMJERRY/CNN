Oui, il existe plusieurs méthodes pour vérifier que les filtres d'un CNN apprennent effectivement. Voici les principales approches :

## 1. **Visualisation des Filtres**
- **Filtres de la première couche** : Visualisez directement les poids des filtres
  ```python
  # Exemple avec Keras/TensorFlow
  first_layer_weights = model.layers[0].get_weights()[0]
  # Afficher les filtres comme des images
  ```
  Les filtres devraient ressembler à des détecteurs de bords (Gabor-like) pour la première couche.

## 2. **Visualisation par Activation Maximisation**
- Trouver l'image d'entrée qui maximise l'activation d'un filtre/channel spécifique
- Montre ce que chaque filtre "recherche"
- Utilisez des outils comme `tf-explain` ou `captum` (pour PyTorch)

## 3. **Feature Maps Visualization**
- Visualisez les sorties (activations) des couches convolutives
  ```python
  from tensorflow.keras.models import Model
  
  # Créer un modèle intermédiaire
  layer_outputs = [layer.output for layer in model.layers]
  activation_model = Model(inputs=model.input, outputs=layer_outputs)
  
  # Obtenir les activations pour une image donnée
  activations = activation_model.predict(test_image)
  ```

## 4. **Analyse des Gradients (Grad-CAM, Guided Backpropagation)**
- **Grad-CAM** : Montre les régions importantes de l'image pour une prédiction
- **Guided Backprop** : Visualise ce que chaque neurone "voit"

## 5. **Analyse Statistique**
- **Suivi de la norme des poids** : Les filtres qui apprennent voient leurs poids évoluer
  ```python
  # Calcul de la norme des filtres par époque
  filter_norms = np.linalg.norm(layer_weights, axis=(0,1,2))
  ```
- **Distribution des activations** : Histogrammes des sorties des filtres

## 6. **Test d'Ablation**
- Désactivez sélectivement des filtres et observez l'impact sur les performances
- Les filtres importants devraient causer une baisse significative d'accuracy

## 7. **Clustering des Feature Maps**
- Regroupez les feature maps similaires pour détecter la redondance
- Les filtres redondants peuvent indiquer un surapprentissage ou un sous-apprentissage

## 8. **Analyse par Projection (t-SNE, UMAP)**
- Projetez les feature maps en 2D pour voir leur diversité
- Une bonne diversité indique que les filtres apprennent des features différentes

## Outils pratiques :
- **TensorBoard** : Visualisation intégrée dans TensorFlow
- **Netron** : Visualisation de l'architecture
- **Captum** (PyTorch) / **tf-explain** (TensorFlow) : Librairies dédiées
- **CNN Explainer** : Outil interactif en ligne

## Signes que les filtres apprennent :
1. **Évolution des poids** pendant l'entraînement
2. **Diversité** dans les feature maps
3. **Spécialisation** progressive (bords → textures → motifs complexes)
4. **Impact mesurable** sur la tâche finale
5. **Interprétabilité** croissante (premières couches = bas niveau, dernières couches = haut niveau)

La combinaison de plusieurs de ces méthodes donne une vision complète de ce que votre CNN apprend réellement.