# Projet Reconnaissance de Formes

## Description

Le projet porte sur la reconnaissance de formes à partir de 9 classes d’objets, avec 11 échantillons par classe. Certaines formes sont occultées ou partiellement représentées, certaines sont distordues, et certaines classes sont hétérogènes.  

L’objectif est d’étudier et de comparer différentes approches classiques de reconnaissance de formes : les **k-plus-proches voisins (KNN, supervisée)**, les **K-Means (non supervisée)**, et le **vote majoritaire** combinant plusieurs descripteurs. Chaque approche est évaluée à partir de vecteurs de représentation extraits via cinq méthodes : **E34, GFD, SA, F0 et F2**, correspondant à différents descripteurs géométriques et fréquentiels des formes.  

Ce projet permet de mettre en évidence les performances relatives de ces méthodes sur une base petite mais complexe, ainsi que les forces et limites des approches supervisées et non supervisées. Il explore également la robustesse des descripteurs face aux variations et distorsions des formes, ainsi que l’impact du choix de paramètres comme le nombre de voisins ou de clusters.

## Techniques et bibliothèques utilisées

- Python 3
- **Modules Python standards** : `collections.Counter`, `numpy`
- **Scikit-learn** pour les métriques : `confusion_matrix`, `classification_report`
- **Matplotlib / Seaborn** pour les visualisations
- Les principales techniques implémentées :  
  - K-Plus Proches Voisins (KNN)  
  - K-Means (nuées dynamiques, non supervisé)  
  - Vote majoritaire combinant plusieurs descripteurs

Aucune installation particulière n’est nécessaire si vous disposez de ces bibliothèques.

## Exécution du script principal
Le fichier `main.py` permet de lancer toutes les analyses et visualisations.
On peut l'exécuter avec la commande : `python main.py`

Note : Le fichier `resultats.txt` contient déjà l’ensemble des résultats générés par `main.py`. Vous pouvez le consulter directement pour éviter de relancer le code plusieurs fois.

## Notebook
Le notebook `notebook.ipynb` permet de visualiser les courbes et graphiques générés, comme la stabilité des modèles, les courbes précision-rappel, ou la comparaison des descripteurs.

## Données
Les données sont dans le dossier `data/` (9 classes, 11 échantillons par classe).

## Structure du projet
```
├── main.py                  # Script principal pour exécuter toutes les tests
├── knn.py                   # Implémentation des méthodes KNN et fonctions associées
├── kmeans.py                # Implémentation K-Means et évaluation de clustering
├── vote_majoritaire.py      # Implémentation du vote majoritaire
├── utils.py                 # Fonctions utilitaires : métriques, filtrage
├── notebooks/               # Notebooks avec visualisations et analyses graphiques
├── data/                    # Données de formes pour les 9 classes
└── README.md                # Ce fichier
```

# Auteurs
Nicolas Adamczyk et Virgil Dinu.