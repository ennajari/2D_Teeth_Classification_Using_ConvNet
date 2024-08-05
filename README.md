# Structure du projet

```plaintext
Copyprojet_classification_dents/
│
├── data/
│   ├── images/
│   └── labels/
│
├── notebooks/
│   ├── 1_exploration_donnees.ipynb
│   ├── 2_pretraitement.ipynb
│   ├── 3_modelisation.ipynb
│   └── 4_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── model.py
│   └── utils.py
│
└── requirements.txt
```

## Description des fichiers Jupyter

### a. 1_exploration_donnees.ipynb

- **Objectif :** Exploration initiale du dataset.
- **Contenu :**
  - Chargement et visualisation des images.
  - Analyse des labels.
  - Statistiques sur le dataset (répartition des classes, dimensions des images, etc.).

#### b. 2_pretraitement.ipynb

- **Objectif :** Prétraitement des données.
- **Contenu :**
  - Redimensionnement des images.
  - Normalisation des images.
  - Augmentation de données (si nécessaire).
  - Division des données en ensembles d'entraînement, de validation et de test.

#### c. 3_modelisation.ipynb

- **Objectif :** Modélisation avec un réseau de neurones convolutifs (ConvNet).
- **Contenu :**
  - Définition de l'architecture du ConvNet.
  - Entraînement du modèle.
  - Visualisation des courbes d'apprentissage (précision, perte).

#### d. 4_evaluation.ipynb

- **Objectif :** Évaluation des performances du modèle.
- **Contenu :**
  - Évaluation du modèle sur l'ensemble de test.
  - Génération et interprétation de la matrice de confusion.
  - Visualisation des prédictions.

### Description des fichiers Python dans le dossier src/

#### a. data_loader.py

- **Objectif :** Chargement et préparation des données.
- **Contenu :**
  - Fonctions pour charger les images et les labels.
  - Prétraitement basique des données (redimensionnement, normalisation).

#### b. preprocess.py

- **Objectif :** Fonctions de prétraitement avancé des images.
- **Contenu :**
  - Fonctions de redimensionnement, normalisation, et augmentation des données.

#### c. model.py

- **Objectif :** Définition et compilation du modèle ConvNet.
- **Contenu :**
  - Définition de l'architecture du réseau.
  - Compilation du modèle avec les hyperparamètres (optimiseur, fonction de perte).

#### d. utils.py

- **Objectif :** Fonctions utilitaires diverses.
- **Contenu :**
  - Fonctions de visualisation des images et des courbes d'apprentissage.
  - Fonctions de calcul des métriques de performance.

### requirements.txt

- **Objectif :** Spécifier les dépendances du projet.
- **Contenu :**
  - Liste des bibliothèques nécessaires (par exemple : numpy, pandas, tensorflow/pytorch, matplotlib, etc.).
