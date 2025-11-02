import os # Pour importer les fichiers de BDShape
import numpy as np # manipuler des listes de nombres sous forme de vecteurs

# Dossier principal contenant les sous-dossiers qui sont les methodes (E34, GFD...), contenant des fichiers contenant les méthodes (de tailles variables)
base_folder = "C:/Users/lakaf/OneDrive/Bureau/25-26/RF/Projet_RF/data"

# Définition des classes, échantillons et méthodes
classes = range(1, 10)  # s01 à s09
samples = range(1, 12)  # n001 à n011
methods = ["E34", "GFD", "SA", "F0", "F2"] # Liste les méthodes des différents descripteurs (chaque methode dans un dossier different)

# Chaque valeurs qu'on a dans les fichiers correspondent a des caractéristiques numériques qui representent la forme de l'image
# L'objectif est de résumer la forme dans un vecteur de nombres (16 valeurs, 128, 100..) qu’on pourra utiliser pour comparer cette forme à d’autres (calcul de distance, classification...)

# Dictionnaire pour stocker toutes les données
data = {}


for c in classes: # On boucle toutes les classes
    for s in samples: # on boucle tous les échantillons pour chaque classe
        key = f"s{c:02d}n{s:03d}"
        data[key] = {"class": c} # Les clés du dictionnaire data sont les classes (E34, GFD...)

        
        for method in methods: # On parcour chaque méthode, qui est enfaite chaque classes (E34, GFD...)
            
            method_folder = os.path.join(base_folder, method)
            filename = os.path.join(method_folder, f"{key}.{method}") # On construit le chemin complet du fichier (ex. ./BDShape/GFD/s01n001.GFD)

            if os.path.exists(filename): # Si le fichier existe
                try:
                    with open(filename, "r") as f:
                        values = f.read().strip().split() # On lit le contenu sous forme de chaine de caracteres (f.read()), on enleve les espace inutiles (f.strip()), on utilise les retour a la ligne comme separateurs (f.split())
                        data[key][method] = np.array(values, dtype=float) # Les values pour une [classe][methde] sera la lecture obtenue converti en tableau NumPy converti en float
                except Exception as e:
                    print(f"Erreur lecture {filename} : {e}") # Si on arrive pas a lire le fichier
            else:
                print(f"Fichier manquant : {filename}") # Si le fichier n'existe pas dans le dossier

print("Lecture terminée.")

print("Nombre total d’images :", len(data))  # Doit être 99 (9 classes x 11 échantillons)
# len(data) renvoie le nombre de clés du dictionnaire data, pas de valeurs, comme pour chaque clé on a une méthode associée, ça donne 99.

exemple = "s01n001"
print("Clés du descripteur :", data[exemple].keys()) # s01n001 est un fichier dans toutes les classes donc ca renvoie chaque classes
for m in methods:
    print(f"{m}: taille = {len(data[exemple][m])}") # Mais dans chaque classes les tailles sont différentes (16, 128, 100...)

# ============================
# Exemple détaillé sur un fichier
# ============================

# Choisir un exemple à inspecter
exemple = "s01n001"

# Vérifier que l'exemple existe dans le dictionnaire
if exemple in data:
    print("\nExemple :", exemple)
    print("Classe réelle :", data[exemple]["class"])
    print("\n--- Détail des descripteurs ---")

    for method in methods:
        if method in data[exemple]:
            desc = data[exemple][method]
            print()
            print(f"Méthode : {method}")
            print(f"Nombre de valeurs : {len(desc)}")
            print(f"Type de données : {type(desc)}")
            print(f"10 premières valeurs :{desc[:10]}")

        else:
            print(f"\nMéthode : {method} (non trouvée pour cette image)")
else:
    print(f"L’exemple {exemple} n’existe pas dans le dictionnaire.")

# ============================
# Implémentation KNN
# ============================

import numpy as np
from collections import Counter # Fonction permettant de compter la fréquence des éléments dans une liste (est-ce qu'on a le droit de l'utiliser ?)

def distance_euclidienne(a, b):
    return np.sqrt(np.sum((a - b) ** 2)) # Distance euclidienne entre deux vecteurs (a et b)

# On prend une image de test (x_test)
# On cherche les k forme d'apprentissage les plus proches d'elle (de x_test) dans l'espace des descripteurs
# On regarde a quelle classe appartiennent ces k formes (la classe majoritaire est celle prédite)
def k_voisins_proches(X_train, y_train, x_test, k=3):
    distances = []
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test)
        distances.append((d, y_train[i]))
    
    # Trier par distance croissante
    distances.sort(key=lambda x: x[0])
    
    # Garder les k plus proches voisins
    voisins = distances[:k]
    
    # Retourner seulement leurs classes
    return [classe for _, classe in voisins]

# On appelle juste la fonction (k_voisins_proches) et on utilise Counter pour savoir quelle classe revient le plus souvent (fréquence d'un element dans une classe)
def predire_classe(X_train, y_train, x_test, k=3):
    voisins = k_voisins_proches(X_train, y_train, x_test, k)
    return Counter(voisins).most_common(1)[0][0]

# X sont les données
# y sont les classes
# Les deux on va les séparer en deux sous ensembles (entrainement et test) : X_train, y_train, X_test, y_test
# ratio à 0.3 = 30% test et 70% train (30% pour les tests)
def separer_train_test(X, y, ratio_test=0.3):
    n = len(X)
    n_test = int(n * ratio_test)
    indices = np.arange(n)
    np.random.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

# ============================
# TEST DU KNN SUR UNE METHODE
# ============================

# On choisit un descripteur, par exemple GFD
X = [] # Valeurs des descripteurs
y = [] # Classes pour chaque descripteurs

for key, value in data.items():
    if "GFD" in value: # Pour chaque image si le descripteur GFD est présent, on ajoute les données dans ces listes
        X.append(value["GFD"])
        y.append(value["class"])

# Conversion en tableau numpy (plus facile a manipuler)
X = np.array(X)
y = np.array(y)

# Séparer en apprentissage et test, il faudra tester plusieurs split
X_train, y_train, X_test, y_test = separer_train_test(X, y, ratio_test=0.3)

# Choisir K
k = 5

# Prédire
y_pred = []
for x in X_test:
    y_pred.append(predire_classe(X_train, y_train, x, k))

y_pred = np.array(y_pred)

# Calculer la précision
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"\n✅ Précision du KNN avec {k} voisins : {accuracy*100:.2f}%")

print("\nExemples de prédictions :")
for i in range(5):
    print(f"Vraie classe : {y_test[i]},  Prédite : {y_pred[i]}")
