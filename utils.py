import numpy as np 
from sklearn.metrics import confusion_matrix, classification_report 

def distance_euclidienne(a, b):
    """
    calcule la distance euclidienne entre deux vecteurs.
    args:
        a, b: np.ndarray de meme dim
    returns: 
        float: dist euclidienne (norme l2)
    """
    # np.sum calcule la somme des carres des differences des elements
    # np.sqrt calcule la racine carree de cette somme
    return np.sqrt(np.sum((a - b) ** 2))

def separer_train_test_par_echantillon(data, echantillons_test, method=None, all_methods=False):
    """
    separe le jeu de donnees en ensembles train/test selon les numeros d'echantillons.
    
    2 cas :
    - si all_methods=true, retourne un dict avec toutes les methodes.
    - sinon, retourne x_train, y_train, x_test, y_test pour la methode donnee.
    
    args:
        data (dict): dict de donnees chargees (ex: {"s01n001": {"class": 1, "e34": vector}, ...}).
        echantillons_test (list): numeros d'echantillons a mettre en test (ex: [1, 2, 3]).
        method (str): methode a utiliser si all_methods=false (ex: "e34").
        all_methods (bool): si true, retourne toutes les methodes.
    returns:
        si all_methods=false: x_train, y_train, x_test, y_test (np.ndarray)
        si all_methods=true: dict {method: {'x_train', 'y_train', 'x_test', 'y_test', 'keys_test'}}
    """
    # liste des methodes dispo
    methods = ["e34", "gfd", "sa", "f0", "f2"]
    
    # fct interne pour effectuer la separation train/test pour une methode donnee
    def _split_for_method(m):
        # init des listes pour les donnees d'entrainement et de test
        x_train, y_train, x_test, y_test, keys_test = [], [], [], [], []
        # parcours de tous les echantillons/images dans le dict 'data'
        for key, value in data.items():
            # verifie si la methode 'm' existe pour cet echantillon
            if m in value:
                # extraction du numero d'echantillon de la cle (ex: "s01n001" -> 001 -> 1)
                sample_num = int(key[4:7]) 
                
                # si le numero fait partie des echantillons de test
                if sample_num in echantillons_test:
                    x_test.append(value[m]) # ajout du vecteur a x_test
                    y_test.append(value["class"]) # ajout de la classe a y_test
                    keys_test.append(key) # ajout de la cle (id echantillon)
                else:
                    # sinon, l'echantillon est mis en entrainement (train)
                    x_train.append(value[m])
                    y_train.append(value["class"])
                    
        # retourne les listes converties en numpy arrays (ou array vide si la liste est vide)
        return {
            'x_train': np.array(x_train) if x_train else np.array([]),
            'y_train': np.array(y_train) if y_train else np.array([]),
            'x_test': np.array(x_test) if x_test else np.array([]),
            'y_test': np.array(y_test) if y_test else np.array([]),
            'keys_test': keys_test
        }
    
    if all_methods:
        # retourne un dict de resultats pour toutes les methodes
        return {m: _split_for_method(m) for m in methods}
    else:
        # retourne les 4 arrays pour la methode specifiee
        split = _split_for_method(method)
        # keys_test est ignore dans ce cas
        return split['x_train'], split['y_train'], split['x_test'], split['y_test']

    
def afficher_matrice_confusion(y_true, y_pred, classes, titre="matrice de confusion"):
    """
    procedure affichant la matrice de confusion avec des entetes pour chaque classe.
    chaque cellule indique le nb de predictions : lignes = classes reelles (vrai positif pour cette classe), colonnes = classes predites (faux positifs si hors diagonale).
    

    args:
        y_true (list): labels reels.
        y_pred (list): labels predits.
        classes (list): liste des classes a afficher.
        titre (str): titre de la matrice affichee.
    """
    # cree la matrice de confusion (lignes = classes reelles, colonnes = classes predites)
    cm = confusion_matrix(y_true, y_pred, labels=classes) 
    print(f"\n{titre} :\n")
    
    # entetes de colonnes (classes predites)
    header = "      " + " ".join([f"{c:>5}" for c in classes]) + " | total"
    print(header)
    print("-" * len(header))
    
    # affichage des lignes (classes reelles) et totaux de lignes
    for i, row in enumerate(cm):
        row_total = np.sum(row) # total des vrais pour cette classe (nb d'echantillons de cette classe)
        row_str = " ".join([f"{val:>5}" for val in row])
        print(f"{classes[i]:>3} | {row_str} | {row_total}")
    
    # total par colonne (nb de fois ou cette classe a ete predite)
    col_totals = np.sum(cm, axis=0)
    col_total_str = " ".join([f"{val:>5}" for val in col_totals])
    print("-" * len(header))
    # affichage des totaux par colonne et du total general (somme de la diagonale ou de tous les elements)
    print(f"tot | {col_total_str} | {np.sum(col_totals)}")

def afficher_classification_report(y_true, y_pred, classes):
    """
    procedure affichant precision, rappel et f1-score pour chaque classe.
    dans notre cas nous n'avons pas de classes absentes, et pas besoin de prioriser le rappel sur la precision (ou inversement). donc le f1-score est une bonne metrique globale.

    args:
        y_true (list): labels reels.
        y_pred (list): labels predits.
        classes (list): liste des classes a afficher.
    """
    # affiche pour chaque classe : precision, rappel et f1-score.
    # on ajoute zero_division=0 pour eviter les erreurs quand une classe n'a pas de prediction (division par zero).
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
