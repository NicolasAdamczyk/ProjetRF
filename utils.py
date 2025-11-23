import os
import numpy as np
from collections import Counter

def charger_donnees(base_folder):
    """
    Charge toutes les données depuis le dossier BDShape
    
    Returns:
        dict: Dictionnaire avec clés 'sXXnYYY' contenant class et méthodes
    """
    classes = range(1, 10)  # s01 à s09
    samples = range(1, 12)  # n001 à n011
    methods = ["E34", "GFD", "SA", "F0", "F2"]
    
    data = {}
    
    for c in classes:
        for s in samples:
            key = f"s{c:02d}n{s:03d}"
            data[key] = {"class": c}
            
            for method in methods:
                method_folder = os.path.join(base_folder, method)
                filename = os.path.join(method_folder, f"{key}.{method}")

                if os.path.exists(filename):
                    try:
                        with open(filename, "r") as f:
                            values = f.read().strip().split()
                            data[key][method] = np.array(values, dtype=float)
                    except Exception as e:
                        print(f"Erreur lecture {filename} : {e}")
                else:
                    print(f"Fichier manquant : {filename}")
    
    print("Lecture terminée.")
    print("Nombre total d'images :", len(data))
    return data

def distance_euclidienne(a, b):
    """Calcule la distance euclidienne entre deux vecteurs"""
    return np.sqrt(np.sum((a - b) ** 2))

def k_voisins_proches_avec_distances(X_train, y_train, x_test, k=3):
    """
    Version améliorée qui retourne aussi les distances pour calculer la confiance
    """
    distances = []
    for i in range(len(X_train)):
        d = distance_euclidienne(X_train[i], x_test)
        distances.append((d, y_train[i]))
    
    # Trier par distance croissante
    distances.sort(key=lambda x: x[0])
    
    # Garder les k plus proches voisins
    return distances[:k]

def k_voisins_proches(X_train, y_train, x_test, k=3):
    """Version simple pour compatibilité"""
    voisins = k_voisins_proches_avec_distances(X_train, y_train, x_test, k)
    return [classe for _, classe in voisins]

def predire_classe(X_train, y_train, x_test, k=3):
    """Prédiction simple sans confiance"""
    voisins = k_voisins_proches(X_train, y_train, x_test, k)
    return Counter(voisins).most_common(1)[0][0]

def predire_classe_avec_confiance(X_train, y_train, x_test, k=3):
    """
    Prédit la classe et calcule un score de confiance
    """
    voisins = k_voisins_proches_avec_distances(X_train, y_train, x_test, k)
    
    # Compter les votes
    classes = [classe for _, classe in voisins]
    vote_counts = Counter(classes)
    
    # Classe majoritaire
    classe_predite = vote_counts.most_common(1)[0][0]
    nb_votes = vote_counts[classe_predite]
    
    # Score de confiance basé sur:
    # 1. Proportion de votes pour la classe gagnante
    # 2. Distance moyenne aux voisins de cette classe
    confiance_vote = nb_votes / k
    
    # Distance moyenne aux voisins de la classe prédite
    distances_classe = [d for d, c in voisins if c == classe_predite]
    if distances_classe:
        distance_moy = np.mean(distances_classe)
        # Normaliser la distance (inverse pour que plus proche = plus confiant)
        confiance_distance = np.exp(-distance_moy / 100)
    else:
        confiance_distance = 0
    
    # Combiner les deux mesures de confiance
    confiance_totale = (confiance_vote + confiance_distance) / 2
    
    return classe_predite, confiance_totale

def separer_train_test_par_echantillon(data, echantillons_test, method=None, all_methods=False):
    """
    Sépare les données en train/test en utilisant des échantillons complets
    pour éviter la fuite de données entre variations similaires.
    
    Args:
        data: dictionnaire de données
        echantillons_test: liste des numéros d'échantillons pour le test
        method: méthode spécifique (si None et all_methods=True, fait toutes les méthodes)
        all_methods: si True, retourne un dict avec toutes les méthodes
    """
    methods = ["E34", "GFD", "SA", "F0", "F2"]
    
    if all_methods:
        data_split = {}
        for m in methods:
            X_train, y_train, X_test, y_test, keys_test = _separer_une_methode(data, m, echantillons_test)
            data_split[m] = {
                'X_train': np.array(X_train) if X_train else np.array([]),
                'y_train': np.array(y_train) if y_train else np.array([]),
                'X_test': np.array(X_test) if X_test else np.array([]),
                'y_test': np.array(y_test) if y_test else np.array([]),
                'keys_test': keys_test
            }
        return data_split
    else:
        X_train, y_train, X_test, y_test, _ = _separer_une_methode(data, method, echantillons_test)
        return (np.array(X_train), np.array(y_train), 
                np.array(X_test), np.array(y_test))

def _separer_une_methode(data, method, echantillons_test):
    """Fonction auxiliaire pour séparer une méthode"""
    X_train, y_train = [], []
    X_test, y_test = [], []
    keys_test = []
    
    for key, value in data.items():
        if method in value:
            # Extraire le numéro d'échantillon (ex: de "s01n005" → 5)
            sample_num = int(key[4:7])
            
            if sample_num in echantillons_test:
                X_test.append(value[method])
                y_test.append(value["class"])
                keys_test.append(key)
            else:
                X_train.append(value[method])
                y_train.append(value["class"])
    
    return X_train, y_train, X_test, y_test, keys_test

def separer_train_test_aleatoire(X, y, ratio_test=0.3):
    """
    ATTENTION : Cette fonction cause une fuite de données !
    Gardée uniquement pour démonstration du problème.
    """
    n = len(X)
    n_test = int(n * ratio_test)
    indices = np.arange(n)
    np.random.shuffle(indices)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def matrice_confusion(y_true, y_pred, n_classes=9):
    """Crée une matrice de confusion"""
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion[true-1, pred-1] += 1
    return confusion

def afficher_matrice_confusion(confusion):
    """Affiche une matrice de confusion formatée"""
    n_classes = confusion.shape[0]
    print("   ", " ".join([f"c{i}" for i in range(1, n_classes+1)]))
    for i in range(n_classes):
        print(f"c{i+1} ", " ".join([f"{confusion[i,j]:2d}" for j in range(n_classes)]))

def analyser_precision_par_classe(y_true, y_pred, n_classes=9):
    """Analyse la précision pour chaque classe"""
    for c in range(1, n_classes+1):
        mask = y_true == c
        if np.sum(mask) > 0:
            acc_classe = np.sum(y_pred[mask] == c) / np.sum(mask)
            print(f"Classe {c}: {acc_classe*100:.2f}% ({np.sum(mask)} échantillons)")

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
