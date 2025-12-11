import numpy as np
from collections import Counter #outil pour compter les occurrences (utilise pour la purete)
from utils import distance_euclidienne

def initialiser_centroides(X, k, method='random'):
    """
    Initialise les centroides pour K-Means.

    Args:
        X (list): données à clusteriser.
        k (int): nombre de clusters.
        method (str): random ou kmeans++ pour l'initialisation.
    Returns:
        np.ndarray: centroides initialisés.
    """
    n = len(X) #nb total de pts
    if method == 'random':
        #init aleatoire: selectionne k points de donnees au hasard comme centroides
        indices = np.random.choice(n, k, replace=False)
        return X[indices].copy() #return les points correspondant a ces indices
    elif method == 'kmeans++':
        #init kmeans++ pour choisir centroides bien separes
        centroids = [] #list centroides selectionnes
        first_idx = np.random.randint(n) #choix aleatoire du 1er centroide 
        centroids.append(X[first_idx])
        for _ in range(1, k):
            #calcul distances minimales a tous les centroides existants (voir utils.py)
            # distance minimale de chaque point x au centroide le plus proche
            distances = np.array([min([distance_euclidienne(x, c) for c in centroids]) for x in X])
            #probabilites de selection (d^2)
            probabilities = distances ** 2
            #normalisation pour avoir une somme de proba egale a 1
            probabilities /= probabilities.sum()
            #selection du prochain centroide avec une proba proportionnelle a distance^2
            next_idx = np.random.choice(n, p=probabilities)
            centroids.append(X[next_idx])
        return np.array(centroids) #retourne la liste de centroides comme un np.array

def assigner_clusters(X, centroides):
    """
    Assigne chaque point de X au cluster le plus proche.

    Args:
        X (list): données
        centroides (list): positions des centroides.
    Returns:
        tuple: (labels des clusters, distances aux centroides assignés)
    """
    n = len(X) #nb de points
    k = len(centroides) #nb de clusters
    labels = np.zeros(n, dtype=int)
    distances = np.zeros(n)
    for i in range(n):
        dists = [distance_euclidienne(X[i], centroides[j]) for j in range(k)] # Calculer distance à chaque centroïde
        labels[i] = np.argmin(dists) # Cluster le plus proche
        distances[i] = min(dists) # Distance au cluster assigné
    return labels, distances

def calculer_centroides(X, labels, k):
    """
    Recalcule les centroides comme la moyenne des points assignés à chaque cluster.

    Args:
        X (list): données.
        labels (list): assignation des clusters.
        k (int): nombre de clusters.
    Returns:
        np.ndarray: nouveaux centroides.
    """
    centroides = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroides[i] = np.mean(cluster_points, axis=0)
        else:
            centroides[i] = X[np.random.randint(len(X))] # Si aucun point dans le cluster, on le réinitialise aléatoirement
    return centroides

def kmeans(X, k, max_iter=100, tol=1e-4, init='kmeans++', n_init=10):
    """
    algorithme k-means avec multi-initialisation.

    args:
        X (list): donnees (np.ndarray).
        k (int): nb de clusters.
        max_iter (int): nb max d'iterations par run.
        tol (float): tolerance pour convergence (changement de centroides negligeable).
        init (str): methode d'init (random ou kmeans++).
        n_init (int): nb d'initialisations aleatoires pour choisir la meilleure (meilleure inertie).

    returns:
        tuple: (labels finaux (np.array), centroides finaux (np.array), inertie finale (float))
    """
    best_inertia = float('inf') # inertie la plus faible trouvee (meilleur score)
    best_labels = None
    best_centroids = None
    
    # multi-initialisation
    for run in range(n_init):
        # init des centroides (random ou kmeans++)
        centroides = initialiser_centroides(X, k, method=init)
        
        # boucle d'iteration de l'algorithme
        for _ in range(max_iter):
            # etape e: assignation des points aux clusters
            labels, distances = assigner_clusters(X, centroides)
            # etape m: recalcul des centroides
            nouveaux_centroides = calculer_centroides(X, labels, k)
            
            # verification de la convergence (les centroides ont-ils bouge de moins que 'tol'?)
            if np.allclose(centroides, nouveaux_centroides, atol=tol): 
                break # si converge, on sort de la boucle d'iteration
            
            centroides = nouveaux_centroides # mise a jour des centroides
        
        # calcul de l'inertie (somme des carres des distances des points a leur centroide)
        inertia = np.sum(distances ** 2)
        
        # mise a jour du meilleur resultat si l'inertie est meilleure
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroides
            
    return best_labels, best_centroids, best_inertia

def evaluer_clustering(y_true, y_pred):
    """
    Évalue la qualité du clustering avec la pureté.

    Args:
        y_true (list): classes réelles.
        y_pred (list): clusters prédits.
    Returns:
        float: pureté du clustering.
    """
    n = len(y_true) # nb total de points
    purity = 0
    # itere sur chaque cluster predit
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster # masque booleen pour les points de ce cluster
        if np.sum(mask) > 0:
            # compte les occurrences des vraies classes dans ce cluster
            # most_common(1)[0][1] recupere le compte de la classe majoritaire
            classe_majoritaire = Counter(y_true[mask]).most_common(1)[0][1]
            purity += classe_majoritaire # ajoute le nb de points bien classes
    # normalise par le nb total de points
    purity /= n
    return purity


def mapper_clusters_vers_classes(y_pred, y_true):
    """
    Associe chaque cluster à la classe majoritaire correspondante.

    Args:
        y_pred (list): clusters prédits.
        y_true (list): classes réelles.
    Returns:
        tuple: (labels mappés selon classes, dictionnaire mapping cluster:classe)
    """
    clusters_uniques = np.unique(y_pred) # liste des indices de clusters (ex: 0, 1, ..., k-1)
    mapping = {} # dict pour stocker l'association cluster -> classe
    for cluster in clusters_uniques:
        mask = y_pred == cluster
        if np.sum(mask) > 0:
            # recupere l'indice (0) de la classe majoritaire ([0][0])
            classe_majoritaire = Counter(y_true[mask]).most_common(1)[0][0] 
            mapping[cluster] = classe_majoritaire # enregistre le mapping
            
    y_mapped = np.zeros_like(y_pred) # array pour les labels apres mapping
    for i in range(len(y_pred)):
        # applique le mapping (cluster -> classe) a chaque point
        y_mapped[i] = mapping[y_pred[i]] 
        
    return y_mapped, mapping

def validation_croisee_kmeans(X, y_true, k=9, n_folds=5, max_iter=100, init="kmeans++", n_init=10):
    """
    Effectue une validation croisée K-Means. (De la même manière que pour KNN).

    Chaque "fold" (pli) consiste à séparer les données en train/test, entraîner K-Means sur le train et évaluer la précision sur le test.

    Args:
        X (list): données.
        y_true (list): classes réelles.
        k (int): nombre de clusters.
        n_folds (int): nombre de plis pour la validation croisée.
        max_iter (int): itérations max K-Means.
        init (str): méthode d'initialisation des centroides.
        n_init (int): nombre d'initialisations K-Means.

    Returns:
        tuple: (accuracy moyenne, écart-type, liste des accuracies par fold)
    """
    X = np.array(X)
    y_true = np.array(y_true)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    fold_size = len(indices) // n_folds
    accuracies = []

    for fold in range(n_folds):
        print(f"\nFold {fold+1}/{n_folds}")

        # Séparer train et test
        test_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_true[train_idx], y_true[test_idx]

        # Entrainer K-Means
        y_train_pred, centroids, _ = kmeans(
            X_train,
            k=k,
            max_iter=max_iter,
            init=init,
            n_init=n_init
        )

        # Mapper clusters vers les classes majoritaires
        _, mapping = mapper_clusters_vers_classes(y_train_pred, y_train)

        # Prédictions sur le test
        y_test_pred = []
        for x in X_test:
            dists = [np.linalg.norm(x - c) for c in centroids]
            cluster = np.argmin(dists)
            y_test_pred.append(mapping.get(cluster, -1)) # par sécurité

        y_test_pred = np.array(y_test_pred)

        acc = np.mean(y_test_pred == y_test)
        accuracies.append(acc)

        print(f"  Accuracy : {acc*100:.2f}%")

    return np.mean(accuracies), np.std(accuracies), accuracies
