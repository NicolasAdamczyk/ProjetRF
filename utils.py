import numpy as np
from sklearn.metrics import confusion_matrix, classification_report # import des fonctions pour évaluer les performances de classification

def distance_euclidienne(a, b):
    """
    Calcule la distance euclidienne entre deux vecteurs.
    Args:
        a, b: np.ndarray de meme dimension
    Returns: 
        float: distance euclidienne
    """
    return np.sqrt(np.sum((a - b) ** 2))

def separer_train_test_par_echantillon(data, echantillons_test, method=None, all_methods=False):
    """
    Sépare le jeu de données en ensembles train/test selon les numéros d'échantillons.
    2 cas :
    - Si all_methods=True, retourne un dictionnaire avec toutes les méthodes.
    - Sinon, retourne X_train, y_train, X_test, y_test pour la méthode donnée.
    
    Args:
        data (dict): dictionnaire de données.
        echantillons_test (list): numéros d'échantillons à mettre en test.
        method (str): méthode à utiliser si all_methods=False.
        all_methods (bool): si True, retourne toutes les méthodes.
    Returns:
        Si all_methods=False: X_train, y_train, X_test, y_test
        Si all_methods=True: dict {method: {'X_train', 'y_train', 'X_test', 'y_test', 'keys_test'}}
    """
    methods = ["E34", "GFD", "SA", "F0", "F2"]
    
    def _split_for_method(m):
        X_train, y_train, X_test, y_test, keys_test = [], [], [], [], []
        for key, value in data.items():
            if m in value:
                sample_num = int(key[4:7])
                if sample_num in echantillons_test:
                    X_test.append(value[m])
                    y_test.append(value["class"])
                    keys_test.append(key)
                else:
                    X_train.append(value[m])
                    y_train.append(value["class"])
        return {
            'X_train': np.array(X_train) if X_train else np.array([]),
            'y_train': np.array(y_train) if y_train else np.array([]),
            'X_test': np.array(X_test) if X_test else np.array([]),
            'y_test': np.array(y_test) if y_test else np.array([]),
            'keys_test': keys_test
        }
    
    if all_methods:
        return {m: _split_for_method(m) for m in methods}
    else:
        split = _split_for_method(method)
        return split['X_train'], split['y_train'], split['X_test'], split['y_test']

    
def afficher_matrice_confusion(y_true, y_pred, classes, titre="Matrice de Confusion"):
    """
    Procédure affichant la matrice de confusion avec des entetes pour chaque classe.
    Chaque cellule indique le nombre de prédictions : lignes = classes réelles (vrai positif pour cette classe), colonnes = classes prédites (faux positifs si hors diagonale).
    
    Args:
        y_true (list): labels réels.
        y_pred (list): labels prédits.
        classes (list): liste des classes à afficher.
        titre (str): titre de la matrice affichée.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes) # Crée la matrice de confusion (lignes = classes réelles, colonnes = classes prédites)
    print(f"\n{titre} :\n")
    
    # Entêtes de colonnes
    header = "      " + " ".join([f"{c:>5}" for c in classes]) + " | Total"
    print(header)
    print("-" * len(header))
    
    # Lignes avec labels de classes et totaux
    for i, row in enumerate(cm):
        row_total = np.sum(row)
        row_str = " ".join([f"{val:>5}" for val in row])
        print(f"{classes[i]:>3} | {row_str} | {row_total}")
    
    # Total par colonne
    col_totals = np.sum(cm, axis=0)
    col_total_str = " ".join([f"{val:>5}" for val in col_totals])
    print("-" * len(header))
    print(f"Tot | {col_total_str} | {np.sum(col_totals)}")

def afficher_classification_report(y_true, y_pred, classes):
    """
    Procédure affichant précision, rappel et F1-score pour chaque classe.
    Dans notre cas nous n'avons pas de classes absentes, et pas besoin de prioriser le rappel sur la précision (ou inversement). Donc le F1-score est une bonne métrique globale.

    Args:
        y_true (list): labels réels.
        y_pred (list): labels prédits.
        classes (list): liste des classes à afficher.
    """
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0)) # Affiche pour chaque classe : précision, rappel et F1-score. On ajoute zero_division=0 pour eviter les erreurs quand une classe n’a pas de prédiction.
