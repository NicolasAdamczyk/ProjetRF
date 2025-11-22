import numpy as np
from collections import Counter
from utils import distance_euclidienne


# ============================
# INITIALISATION DES CENTROIDES
# ============================
def initialiser_centroides(X, k, method='random'):
    n = len(X)
    if method == 'random':
        indices = np.random.choice(n, k, replace=False)
        return X[indices].copy()
    elif method == 'kmeans++':
        centroids = []
        first_idx = np.random.randint(n)
        centroids.append(X[first_idx])
        for _ in range(1, k):
            distances = np.array([min([distance_euclidienne(x, c) for c in centroids]) for x in X])
            probabilities = distances ** 2
            probabilities /= probabilities.sum()
            next_idx = np.random.choice(n, p=probabilities)
            centroids.append(X[next_idx])
        return np.array(centroids)


# ============================
# ASSIGNATION DES POINTS AUX CLUSTERS
# ============================
def assigner_clusters(X, centroides):
    n = len(X)
    k = len(centroides)
    labels = np.zeros(n, dtype=int)
    distances = np.zeros(n)
    for i in range(n):
        dists = [distance_euclidienne(X[i], centroides[j]) for j in range(k)]
        labels[i] = np.argmin(dists)
        distances[i] = min(dists)
    return labels, distances


# ============================
# RECALCUL DES CENTROIDES
# ============================
def calculer_centroides(X, labels, k):
    centroides = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroides[i] = np.mean(cluster_points, axis=0)
        else:
            centroides[i] = X[np.random.randint(len(X))]
    return centroides


# ============================
# K-MEANS COMPLET
# ============================
def kmeans(X, k, max_iter=100, tol=1e-4, init='kmeans++', n_init=10):
    best_inertia = float('inf')
    best_labels = None
    best_centroids = None
    for run in range(n_init):
        centroides = initialiser_centroides(X, k, method=init)
        for _ in range(max_iter):
            labels, distances = assigner_clusters(X, centroides)
            nouveaux_centroides = calculer_centroides(X, labels, k)
            if np.allclose(centroides, nouveaux_centroides, atol=tol):
                break
            centroides = nouveaux_centroides
        inertia = np.sum(distances ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroides
    return best_labels, best_centroids, best_inertia


# ============================
# EVALUATION DU CLUSTERING
# ============================
def evaluer_clustering(y_true, y_pred):
    n = len(y_true)
    k = len(np.unique(y_pred))
    # Pureté
    purity = 0
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        if np.sum(mask) > 0:
            classe_majoritaire = Counter(y_true[mask]).most_common(1)[0][1]
            purity += classe_majoritaire
    purity /= n

    # Matrice de confusion pour le clustering
    confusion = np.zeros((k, 9), dtype=int)
    for i in range(n):
        confusion[y_pred[i], y_true[i] - 1] += 1
    return purity, confusion


# ============================
# MAPPER CLUSTERS VERS CLASSES
# ============================
def mapper_clusters_vers_classes(y_pred, y_true):
    clusters_uniques = np.unique(y_pred)
    mapping = {}
    for cluster in clusters_uniques:
        mask = y_pred == cluster
        if np.sum(mask) > 0:
            classe_majoritaire = Counter(y_true[mask]).most_common(1)[0][0]
            mapping[cluster] = classe_majoritaire
    y_mapped = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_mapped[i] = mapping[y_pred[i]]
    return y_mapped, mapping
