import os
import numpy as np
from collections import Counter

# Fixer la seed pour la reproductibilité, pour que les tests soient identiques
np.random.seed(42)

# Dossier principal contenant les sous-dossiers qui sont les methodes
base_folder = "./data"

# Définition des classes, échantillons et méthodes
classes = range(1, 10)  # s01 à s09
samples = range(1, 12)  # n001 à n011
methods = ["E34", "GFD", "SA", "F0", "F2"]

# Dictionnaire pour stocker toutes les données
data = {}

# Lecture des données
for c in classes: # pour chaque classe et échantillon
    for s in samples:
        key = f"s{c:02d}n{s:03d}" # on crée une clé unique
        data[key] = {"class": c}
        
        for method in methods:
            method_folder = os.path.join(base_folder, method)
            filename = os.path.join(method_folder, f"{key}.{method}") # construction du chemin vers le fichier

            if os.path.exists(filename):
                try:
                    with open(filename, "r") as f:
                        values = f.read().strip().split()
                        data[key][method] = np.array(values, dtype=float) # conversion en numpy array float
                except Exception as e:
                    print(f"Erreur lecture {filename} : {e}")
            else:
                print(f"Fichier manquant : {filename}")

print("Lecture terminée.")
print("Nombre total d'images :", len(data))

# ============================
# Implémentation KNN
# ============================

# Distance euclidienne entre deux vecteurs a et b
# Cela servira à voir quelles formes sont proches dans KNN
def distance_euclidienne(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Trouver les k plus proches voisins
def k_voisins_proches(X_train, y_train, x_test, k=3):
    distances = []
    # Calculer la distance entre chaque point de train et le test
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test)
        distances.append((d, y_train[i]))
    
    # Trier par distance pour garder les k plus proches voisins
    distances.sort(key=lambda x: x[0])
    
    # Garder les k plus proches voisins
    voisins = distances[:k]
    
    # Retourner seulement leurs classes
    return [classe for _, classe in voisins]

#
def predire_classe(X_train, y_train, x_test, k=3):
    voisins = k_voisins_proches(X_train, y_train, x_test, k)

    '''
    # Si jamais on peut pas utiliser Counter, la solution :
        # Compter combien de fois chaque classe apparaît
    counts = {}
    for v in voisins:
        counts[v] = counts.get(v, 0) + 1

    # Trouver la classe la plus fréquente
    classe_predite = max(counts, key=counts.get)
    return classe_predite
    '''

    return Counter(voisins).most_common(1)[0][0] # Utilisation de Counter pour choisir la classe majoritaire.

# ============================
# ANCIENNE FONCTION (PROBLÉMATIQUE)
# ============================

def separer_train_test_ANCIENNE(X, y, ratio_test=0.3):
    """
    ATTENTION : Cette fonction cause une fuite de données !
    Elle mélange tous les échantillons sans tenir compte de leur similarité.
    """
    n = len(X)
    n_test = int(n * ratio_test)
    indices = np.arange(n)
    np.random.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

# ============================
# NOUVELLE FONCTION DE SÉPARATION
# ============================

def separer_train_test_par_echantillon(data, method, echantillons_test):
    """
    Sépare les données en train/test en utilisant des échantillons complets
    pour éviter la fuite de données entre variations similaires.
    
    Par exemple, si echantillons_test = [1, 5, 9], alors tous les n001, n005, n009
    de toutes les classes iront dans le test.
    """
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for key, value in data.items():
        if method in value:
            # Extraire le numéro d'échantillon (ex: de "s01n005" → 5)
            sample_num = int(key[4:7])
            
            if sample_num in echantillons_test:
                X_test.append(value[method])
                y_test.append(value["class"])
            else:
                X_train.append(value[method])
                y_train.append(value["class"])
    
    return (np.array(X_train), np.array(y_train), 
            np.array(X_test), np.array(y_test))

# ============================
# VALIDATION CROISÉE
# ============================

def validation_croisee_knn(data, method, k=3, n_folds=3):
    """
    Effectue une validation croisée en séparant les échantillons
    de manière à éviter la fuite de données.
    """
    # Créer les folds (groupes d'échantillons)
    echantillons = list(range(1, 12))
    np.random.shuffle(echantillons)
    
    # Diviser en n_folds groupes
    fold_size = len(echantillons) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            # Le dernier fold prend les échantillons restants
            folds.append(echantillons[start:])
        else:
            folds.append(echantillons[start:start + fold_size])
    
    accuracies = []
    
    for i, test_samples in enumerate(folds):
        print(f"\nFold {i+1}/{n_folds}")
        print(f"Échantillons de test : {test_samples}")
        
        # Séparer les données
        X_train, y_train, X_test, y_test = separer_train_test_par_echantillon(
            data, method, test_samples
        )
        
        print(f"Taille train : {len(X_train)}, Taille test : {len(X_test)}")
        
        # Prédire
        y_pred = []
        for x in X_test:
            y_pred.append(predire_classe(X_train, y_train, x, k))
        
        y_pred = np.array(y_pred)
        
        # Calculer la précision
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        accuracies.append(accuracy)
        
        print(f"Précision : {accuracy*100:.2f}%")
        
        # Afficher quelques exemples
        print("Exemples de prédictions :")
        for j in range(min(3, len(y_test))):
            print(f"  Vraie classe : {y_test[j]}, Prédite : {y_pred[j]}")
    
    # Moyenne et écart-type
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    return mean_acc, std_acc, accuracies

# ============================
# DÉMONSTRATION DU PROBLÈME
# ============================

print("="*50)
print("DÉMONSTRATION DU PROBLÈME DE FUITE DE DONNÉES")
print("="*50)

# Préparer les données avec GFD
X_old = []
y_old = []
for key, value in data.items():
    if "GFD" in value:
        X_old.append(value["GFD"])
        y_old.append(value["class"])

X_old = np.array(X_old)
y_old = np.array(y_old)

# Test avec l'ancienne méthode (problématique)
print("\n1. ANCIENNE MÉTHODE (avec fuite de données) :")
X_train_old, y_train_old, X_test_old, y_test_old = separer_train_test_ANCIENNE(X_old, y_old, ratio_test=0.3)

y_pred_old = []
for x in X_test_old:
    y_pred_old.append(predire_classe(X_train_old, y_train_old, x, k=5))

accuracy_old = np.sum(np.array(y_pred_old) == y_test_old) / len(y_test_old)
print(f"Précision : {accuracy_old*100:.2f}%")
print("⚠️ Cette précision est artificiellement élevée !")

# ============================
# TEST DU KNN CORRIGÉ
# ============================

print("\n\n" + "="*50)
print("TEST KNN AVEC SÉPARATION CORRIGÉE")
print("="*50)

# Test avec différentes valeurs de k
for method in ["GFD", "E34"]:
    print(f"\n\nMÉTHODE : {method}")
    print("-"*30)
    
    for k in [3, 5, 7]:
        print(f"\nTest avec k={k} voisins")
        mean_acc, std_acc, accuracies = validation_croisee_knn(
            data, method, k=k, n_folds=3
        )
        
        print(f"\nRésumé pour k={k}:")
        print(f"Précisions par fold : {[f'{acc*100:.2f}%' for acc in accuracies]}")
        print(f"Précision moyenne : {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")

# ============================
# ANALYSE DES ERREURS
# ============================

print("\n\n" + "="*50)
print("ANALYSE DÉTAILLÉE")
print("="*50)

# Créer une matrice de confusion simple
def matrice_confusion_simple(y_true, y_pred, n_classes=9):
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion[true-1, pred-1] += 1
    return confusion

# Test sur un fold pour analyser les erreurs
X_train, y_train, X_test, y_test = separer_train_test_par_echantillon(
    data, "GFD", [2, 5, 8]
)

y_pred = []
for x in X_test:
    y_pred.append(predire_classe(X_train, y_train, x, k=5))

y_pred = np.array(y_pred)

print("\nMatrice de confusion (lignes=vraies classes, colonnes=prédictions):")
confusion = matrice_confusion_simple(y_test, y_pred)
print("   ", " ".join([f"c{i}" for i in range(1, 10)]))
for i in range(9):
    print(f"c{i+1} ", " ".join([f"{confusion[i,j]:2d}" for j in range(9)]))

# Analyser les erreurs par classe
print("\nPrécision par classe:")
for c in range(1, 10):
    mask = y_test == c
    if np.sum(mask) > 0:
        acc_classe = np.sum(y_pred[mask] == c) / np.sum(mask)
        print(f"Classe {c}: {acc_classe*100:.2f}% ({np.sum(mask)} échantillons)")
