
import numpy as np
from collections import Counter 
from utils import distance_euclidienne 

def k_voisins_proches(X_train, y_train, x_test, k=3):
    """
    renvoie les classes des k plus proches voisins d'un point test. (etape 1 du knn)

    args:
        X_train (list) : donnees d'entrainement. (np.ndarray ou list de vecteurs)
        y_train (list) : labels d'entrainement. (np.ndarray ou list d'int/str)
        x_test (list) : vecteur a classer. (vecteur unique)
        k (int) : nb de voisins.
    returns:
        list : classes des k voisins les plus proches.
    """
    distances = []
    # calcul de la distance entre x_test et chaque point de x_train
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test) # calcul de la distance euclidienne (voir utils.py)
        distances.append((d, y_train[i])) # stocke (distance, classe)
        
    distances.sort(key=lambda x: x[0]) # tri par distance croissante (x[0] est la distance)
    voisins = distances[:k] # selection des k plus proches
    # retourne uniquement la classe (x[1]) pour les k voisins selectionnes
    return [classe for _, classe in voisins]


def k_voisins_proches_avec_distances(X_train, y_train, x_test, k=3):
    """
    version amelioree de k_voisins_proches qui retourne les distances.

    args:
        X_train (list) : donnees d'entrainement.
        y_train (list) : labels d'entrainement.
        x_test (list) : vecteur a classer.
        k (int) : nb de voisins.
    returns:
        list of tuples: [(distance, classe), ...] des k voisins les plus proches.
    """
    distances = []
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test)
        distances.append((d, y_train[i])) # stocke (distance, classe)
    
    # tri par distance croissante
    distances.sort(key=lambda x: x[0])
    
    # garde les k p.p.v (distance et classe)
    return distances[:k]

def predire_knn(X_train, y_train, x_test, k=3):
    """
    predit la classe majoritaire parmi les k voisins les plus proches. (etape 2 du knn)
    

    args:
        X_train (list) : donnees d'entrainement.
        y_train (list) : labels d'entrainement.
        x_test (list) : vecteur a classer.
        k (int) : nb de voisins.
    returns:
        int: classe predite (majoritaire).
    """
    voisins = k_voisins_proches(X_train, y_train, x_test, k) # recupere les classes des k voisins

    # returne la classe la plus frequente parmi les voisins (vote majoritaire)
    # most_common(1)[0][0] recupere l'element (la classe) ayant le compte max
    return Counter(voisins).most_common(1)[0][0]

def predire_classe_avec_confiance(X_train, y_train, x_test, k=3):
    """
    predit la classe d'un point et calcule un score de confiance base sur le vote et la distance.
    elle peut etre utilisee directement pour un vote majoritaire (voir vote_majoritaire.py), pour combiner plusieurs predictions ou methodes.

    args:
        X_train (list) : donnees d'entrainement.
        y_train (list) : labels d'entrainement.
        x_test (list) : vecteur a classer.
        k (int) : nb de voisins.
    returns:
        tuple: (classe predite (int/str), score de confiance [0-1] (float))
    """
    voisins = k_voisins_proches_avec_distances(X_train, y_train, x_test, k) # (distance, classe) des voisins
    
    # comptage des votes
    classes = [classe for _, classe in voisins]
    vote_counts = Counter(classes)
    
    # classe majoritaire et nombre de votes
    classe_predite = vote_counts.most_common(1)[0][0]
    nb_votes = vote_counts[classe_predite]
    
    # confiance basee sur la proportion de votes (max 1.0 si tous votent pour la meme classe)
    confiance_vote = nb_votes / k
    
    # confiance basee sur la distance aux voisins de la classe predite
    distances_classe = [d for d, c in voisins if c == classe_predite]
    if distances_classe:
        distance_moy = np.mean(distances_classe)
        # utilise une fonction d'attenuation (ici exp negative) : distance plus faible = confiance plus elevee
        # l'argument 100 est arbitraire pour la mise a l'echelle
        confiance_distance = np.exp(-distance_moy / 100) # plus proche = plus confiant
    else:
        confiance_distance = 0
    
    # combiner les deux mesures de confiance (moyenne simple ici)
    confiance_totale = (confiance_vote + confiance_distance) / 2
    
    return classe_predite, confiance_totale

def validation_croisee_knn(X, y, k=3, n_folds=5):
    """
    effectue une validation croisee pour knn. elle consiste a diviser les donnees en n_folds (parties en francais), puis a entrainer et tester le modele knn sur chaque fold, puis enfin evaluer la precision.
    pour chaque fold :
        - les donnees de ce fold servent de test,
        - les donnees des autres folds servent d'entrainement.
    

    args:
        X (list): donnees. (list de vecteurs)
        y (list): labels correspondants. (list de labels)
        k (int): nb de voisins pour knn.
        n_folds (int): nb de folds pour la validation croisee.
    returns:
        tuple: (precision moyenne (float), ecart-type (float), liste des accuracies par fold (list de float))
    """
    # conversion en np.array si ce n'est pas deja fait (necessaire pour np.arange et np.setdiff1d)
    X = np.array(X) 
    y = np.array(y)
    
    indices = np.arange(len(X)) # creation des indices [0, 1, ..., n-1]
    np.random.shuffle(indices) # melange aleatoire pour creer des folds aleatoires

    fold_size = len(indices) // n_folds # taille de chaque pli
    accuracies = [] # liste pour stocker les resultats

    for i in range(n_folds):
        # separation train/test:
        # selection des indices du pli actuel pour le test
        test_idx = indices[i*fold_size : (i+1)*fold_size]
        # tous les autres indices pour l'entrainement
        train_idx = np.setdiff1d(indices, test_idx)

        # extraction des donnees (utilisation des listes d'indices sur les np.array)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test  = X[test_idx]
        y_test  = y[test_idx]

        # prediction sur l'ensemble de test
        y_pred = np.array([
            predire_knn(X_train, y_train, x, k) # applique predire_knn a chaque point x de x_test
            for x in X_test
        ])

        # calcul de la precision (accuracy)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)

        print(f"fold {i+1}/{n_folds} – accuracy : {acc*100:.2f}%")

    # retourne la moyenne et l'ecart-type des accuracies
    return np.mean(accuracies), np.std(accuracies), accuracies
