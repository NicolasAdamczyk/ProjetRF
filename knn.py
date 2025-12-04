import numpy as np
from collections import Counter # Import de Counter pour compter rapidement les occurrences d'éléments dans une liste (utile pour déterminer la classe majoritaire parmi les voisins)
from utils import distance_euclidienne

def k_voisins_proches(X_train, y_train, x_test, k=3):
    """
    Renvoie les classes des k plus proches voisins d'un point test.

    Args:
        X_train (list) : données d'entraînement.
        y_train (list) : labels d'entraînement.
        x_test (list) : vecteur à classer.
        k (int) : nombre de voisins.
    Returns:
        list : classes des k voisins les plus proches.
    """
    distances = []
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test) # Calcul de la distance euclidienne (voir utils.py)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0]) # Tri par distance croissante
    voisins = distances[:k] # Sélection des k plus proches
    return [classe for _, classe in voisins]


def k_voisins_proches_avec_distances(X_train, y_train, x_test, k=3):
    """
    Version améliorée de k_voisins_proches qui retourne les distances.

    Args:
        X_train (list) : données d'entraînement.
        y_train (list) : labels d'entraînement.
        x_test (list) : vecteur à classer.
        k (int) : nombre de voisins.
    Returns:
        list of tuples: [(distance, classe), ...] des k voisins les plus proches.
    """
    distances = []
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test)
        distances.append((d, y_train[i]))
    
    # Tri par distance croissante
    distances.sort(key=lambda x: x[0])
    
    # Garder les distances des k plus proches voisins
    return distances[:k]

def predire_knn(X_train, y_train, x_test, k=3):
    """
    Prédit la classe majoritaire parmi les k voisins les plus proches.

    Args:
        X_train (list) : données d'entraînement.
        y_train (list) : labels d'entraînement.
        x_test (list) : vecteur à classer.
        k (int) : nombre de voisins.
    Returns:
        int: classe prédite.
    """
    voisins = k_voisins_proches(X_train, y_train, x_test, k)

    # Version sans Counter (si l'on doit éviter cette dépendance)
    # counts = {}
    # for v in voisins:
    #     counts[v] = counts.get(v, 0) + 1
    # return max(counts, key=counts.get)

    return Counter(voisins).most_common(1)[0][0]

def predire_classe_avec_confiance(X_train, y_train, x_test, k=3):
    """
    Prédit la classe d'un point et calcule un score de confiance basé sur le vote et la distance.
    Elle peut être utilisée directement pour un vote majoritaire (voir vote_majoritaire.py), pour combiner plusieurs prédictions ou méthodes.

    Args:
        X_train (list) : données d'entraînement.
        y_train (list) : labels d'entraînement.
        x_test (list) : vecteur à classer.
        k (int) : nombre de voisins.
    Returns:
        tuple: (classe prédite, score de confiance [0-1])
    """
    voisins = k_voisins_proches_avec_distances(X_train, y_train, x_test, k)
    
    # Comptage des votes
    classes = [classe for _, classe in voisins]
    vote_counts = Counter(classes)
    
    # Classe majoritaire et nombre de votes
    classe_predite = vote_counts.most_common(1)[0][0]
    nb_votes = vote_counts[classe_predite]
    
    # Confiance basée sur la proportion de votes
    confiance_vote = nb_votes / k
    
    # Confiance basée sur la distance aux voisins de la classe prédite
    distances_classe = [d for d, c in voisins if c == classe_predite]
    if distances_classe:
        distance_moy = np.mean(distances_classe)
        confiance_distance = np.exp(-distance_moy / 100) # plus proche = plus confiant
    else:
        confiance_distance = 0
    
    # Combiner les deux mesures de confiance
    confiance_totale = (confiance_vote + confiance_distance) / 2
    
    return classe_predite, confiance_totale

def validation_croisee_knn(X, y, k=3, n_folds=5):
    """
    Effectue une validation croisée pour KNN. Elle consiste à diviser les données en n_folds (parties en français), puis à entraîner et tester le modèle KNN sur chaque fold, puis enfin évaluer la précision.
    Pour chaque fold :
        - les données de ce fold servent de test,
        - les données des autres folds servent d'entraînement.

    Args:
        X (list): données.
        y (list): labels correspondants.
        k (int): nombre de voisins pour KNN.
        n_folds (int): nombre de folds pour la validation croisée.
    Returns:
        tuple: (précision moyenne, écart-type, liste des accuracies par fold)
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    fold_size = len(indices) // n_folds
    accuracies = []

    for i in range(n_folds):
        test_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        # Séparer train/test
        X_train = [X[j] for j in train_idx]
        y_train = [y[j] for j in train_idx]
        X_test  = [X[j] for j in test_idx]
        y_test  = [y[j] for j in test_idx]

        # Prédiction
        y_pred = np.array([
            predire_knn(X_train, y_train, x, k) 
            for x in X_test
        ])

        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)

        print(f"Fold {i+1}/{n_folds} – Accuracy : {acc*100:.2f}%")

    return np.mean(accuracies), np.std(accuracies), accuracies

