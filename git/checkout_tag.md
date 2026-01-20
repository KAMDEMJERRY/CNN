Pour récupérer une version particulière taguée, voici les différentes méthodes :

## 1. **Voir les tags disponibles**
```bash
git tag -l
git tag -l "v*"  # Filtrer par pattern
```

## 2. **Méthodes pour récupérer une version**

### **A. Créer une branche à partir d'un tag (recommandé pour travailler)**
```bash
# Créer une nouvelle branche à partir du tag
git checkout -b ma-version-stable v1.0.0

# Ou basculer directement sur le tag
git checkout v1.0.0
# ⚠️ Vous serez en mode "detached HEAD" - créez une branche pour modifier
```

### **B. Récupérer une version spécifique sans toucher au workspace actuel**
```bash
# Voir le contenu d'un tag
git show v1.0.0

# Comparer deux versions
git diff v1.0.0 v1.1.0

# Voir les fichiers d'une version
git ls-tree -r v1.0.0
```

### **C. Récupérer un fichier spécifique d'une version**
```bash
# Récupérer un fichier précis
git show v1.0.0:chemin/vers/mon-modele.py > modele-v1.0.0.py

# Récupérer un dossier
git archive v1.0.0 --output=version-1.0.0.zip
```

### **D. Cloner une version spécifique**
```bash
# Cloner le repo puis checkout le tag
git clone https://github.com/votre-repo.git
cd votre-repo
git checkout v1.0.0

# Ou cloner directement un tag (profondeur 1 pour économiser de l'espace)
git clone --branch v1.0.0 --depth 1 https://github.com/votre-repo.git
```

## 3. **Pour les modèles ML - scénarios courants**

### **Récupérer les poids/checkpoints**
```bash
# Si vos checkpoints sont versionnés
git checkout v1.0.0 -- model_checkpoints/
```

### **Créer un environnement reproductible**
```bash
# 1. Récupérer la version du code
git checkout v1.0.0

# 2. Vérifier les dépendances dans requirements.txt ou setup.py
cat requirements.txt

# 3. Installer les dépendances exactes
pip install -r requirements.txt
```

## 4. **Trouver quel tag contient un fichier/changement spécifique**
```bash
# Voir quel tag contient un commit
git describe --tags [commit-hash]

# Chercher un changement dans tous les tags
git log --oneline --decorate --tags --grep="fix"  # Chercher "fix" dans les messages
```

## 5. **Si vous avez besoin de modifier une ancienne version**
```bash
# 1. Créer une branche hotfix à partir de l'ancienne version
git checkout -b hotfix-v1.0.1 v1.0.0

# 2. Faire vos modifications
# 3. Créer un nouveau tag
git tag -a v1.0.1 -m "Correctif pour la version 1.0.0"

# 4. Fusionner si nécessaire dans la branche principale
git checkout main
git merge hotfix-v1.0.1
```

## 6. **Exemple concret pour un modèle ML**
```bash
# Récupérer la version 1.0 de mon modèle
git checkout v1.0.0

# Installer les dépendances exactes
pip install -r requirements.txt

# Vérifier la configuration du modèle
cat config/model_config.yaml

# Tester le modèle
python test_model.py --version v1.0.0
```

## **Conseils importants :**
- Les tags sont immuables dans Git
- Utilisez `git checkout -b [nom-branche] [tag]` pour éviter le "detached HEAD"
- Pour les modèles, versionnez aussi :
  - Les poids du modèle (`.pth`, `.h5`, etc.)
  - La configuration d'entraînement
  - Le fichier `requirements.txt` avec les versions exactes
  - Un script de validation/reproduction

Vous cherchez à récupérer une version spécifique pour quel usage particulier ?