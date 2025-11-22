import numpy as np
from collections import Counter
from utils import distance_euclidienne


def k_voisins_proches(X_train, y_train, x_test, k=3):
    """Renvoie les classes des k plus proches voisins du point x_test."""
    distances = []
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])
    voisins = distances[:k]
    return [classe for _, classe in voisins]

def predire_knn(X_train, y_train, x_test, k=3):
    """Prédit la classe majoritaire parmi les k voisins les plus proches."""
    voisins = k_voisins_proches(X_train, y_train, x_test, k)

    # Version sans Counter (au cas où tu veux éviter cette dépendance)
    # counts = {}
    # for v in voisins:
    #     counts[v] = counts.get(v, 0) + 1
    # return max(counts, key=counts.get)

    return Counter(voisins).most_common(1)[0][0]

def validation_croisee_knn(X, y, k=3, n_folds=5):
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

