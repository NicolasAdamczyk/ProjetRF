import numpy as np
from utils import separer_train_test_par_echantillon # Utilise la version all_methods=True
from knn import predire_classe_avec_confiance # Utilise de la version de prédication avec confiance

methods = ["E34", "GFD", "SA", "F0", "F2"] # Pour les itérations sur les méthodes disponibles

class VoteMajoritaire:
    """
    Classifieur par vote majoritaire combinant plusieurs méthodes KNN.

    Args:
        k (int): nombre de voisins pour KNN.
        weights (dict): poids de chaque méthode. Si None, poids = 1.0 pour toutes.
        use_confidence (bool): si True, pondère le vote par la confiance des KNN.
    """

    def __init__(self, k=5, weights=None, use_confidence=True):
        """
        Constructeur initialisant le classifieur VoteMajoritaire.
        """
        self.k = k
        self.use_confidence = use_confidence
        self.weights = weights if weights else {m: 1.0 for m in methods}
        self.data_train = {} # stocke X_train et y_train par méthode

    def entrainer(self, data_split, verbose=True):
        """
        Stocke les données d'entraînement pour chaque méthode disponible.

        Args:
            data_split (dict): données séparées train/test pour chaque méthode.
            verbose (bool): si True, affiche le nombre de méthodes entraînées.
        """
        self.data_train = {}
        for method in methods:
            if method in data_split:
                self.data_train[method] = {
                    'X': data_split[method]['X_train'],
                    'y': data_split[method]['y_train']
                }
        if verbose:
            print(f"Entraînement terminé avec {len(self.data_train)} méthodes")

    def predire_ensemble(self, data_split, mode="majoritaire_simple"):
        """
        Prédit les classes pour l'ensemble test en combinant les votes des différentes méthodes.

        Args:
            data_split (dict): données séparées train/test.
            mode (str): 3 stratégies de vote:
                - "majoritaire_simple": un vote par méthode
                - "majoritaire_pondere": vote pondéré par poids
                - "majoritaire_confiance": vote pondéré par confiance et poids
        Returns:
            tuple: (np.ndarray: classes prédites finales, list: détails des votes pour chaque échantillon)
        """
        first_method = list(data_split.keys())[0]
        n_test = len(data_split[first_method]['X_test'])
        y_test = data_split[first_method]['y_test']
        
        predictions_finales = []
        details_votes = []
        
        for i in range(n_test):
            votes = {}
            predictions_par_methode = {}
            
            for method in methods:
                if method in self.data_train and len(self.data_train[method]['X']) > 0:
                    x_test = data_split[method]['X_test'][i]
                    
                    # Prédiction KNN avec score de confiance
                    classe, confiance = predire_classe_avec_confiance(
                        self.data_train[method]['X'],
                        self.data_train[method]['y'],
                        x_test,
                        self.k
                    )
                    
                    predictions_par_methode[method] = classe
                    
                    if classe not in votes:
                        votes[classe] = 0
                    
                    # Accumulation des votes selon le mode choisi
                    if mode == 'majoritaire_simple':
                        votes[classe] += 1
                    elif mode == 'majoritaire_pondere':
                        votes[classe] += self.weights[method]
                    elif mode == 'majoritaire_confiance':
                        votes[classe] += confiance * self.weights[method]

            # Classe avec le maximum de votes
            if votes:
                classe_finale = max(votes, key=votes.get)
            else:
                classe_finale = 1
            
            predictions_finales.append(classe_finale)
            details_votes.append({
                'predictions': predictions_par_methode,
                'votes': votes,
                'vraie_classe': y_test[i],
                'classe_predite': classe_finale
            })
        
        return np.array(predictions_finales), details_votes


# ============================================================
# ÉVALUATION DES DESCRIPTEURS INDIVIDUELS
# ============================================================

def evaluer_methodes_individuelles(data_split, k=5):
    """
    Évalue chaque méthode individuellement avec KNN.

    Args:
        data_split (dict): données séparées train/test pour chaque méthode.
        k (int): nombre de voisins KNN.

    Returns:
        dict: résultats par méthode avec 'accuracy', 'y_pred' et 'y_test'.
    """
    resultats = {}
    
    for method in methods:
        if method not in data_split or len(data_split[method]['X_train']) == 0:
            continue
            
        X_train = data_split[method]['X_train']
        y_train = data_split[method]['y_train']
        X_test = data_split[method]['X_test']
        y_test = data_split[method]['y_test']
        
        y_pred = []
        for x in X_test:
            pred, _ = predire_classe_avec_confiance(X_train, y_train, x, k)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        accuracy = np.sum(y_pred == y_test) / len(y_test) if len(y_test) > 0 else 0
        
        resultats[method] = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    return resultats

def filtrer_donnees_classes(data, classes_a_garder):
    """
    Filtre les données pour ne garder que certaines classes.

    Args:
        data (dict): dictionnaire de données original.
        classes_a_garder (list): classes à conserver.

    Returns:
        dict: dictionnaire filtré.
    """
    data_filtre = {}
    for key, value in data.items():
        if value["class"] in classes_a_garder:
            data_filtre[key] = value
    return data_filtre

def validation_croisee_complete(data, k=5, n_folds=3, n_classes=9):
    """
    Effectue une validation croisée complète combinant toutes les méthodes et tous les modes de vote.

    Args:
        data (dict): données originales.
        k (int): nombre de voisins KNN.
        n_folds (int): nombre de plis (folds) pour la validation croisée.
        n_classes (int): nombre de classes dans le dataset.

    Returns:
        dict: dictionnaire contenant:
            - resultats_par_mode
            - resultats_individuels
            - all_y_true / all_y_pred
            - all_y_true_ind / all_y_pred_ind
            - best_mode
            - n_classes
    """
    echantillons = list(range(1, 12))
    np.random.shuffle(echantillons)
    
    fold_size = len(echantillons) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            folds.append(echantillons[start:])
        else:
            folds.append(echantillons[start:start + fold_size])
    
    modes = ['majoritaire_simple', 'majoritaire_pondere', 'majoritaire_confiance']
    resultats_par_mode = {mode: [] for mode in modes}
    resultats_individuels = {method: [] for method in methods}
    
    # Accumuler toutes les prédictions pour la matrice de confusion
    all_y_true = {mode: [] for mode in modes}
    all_y_pred = {mode: [] for mode in modes}
    all_y_true_ind = {m: [] for m in methods}
    all_y_pred_ind = {m: [] for m in methods}
    
    poids_personnalise = {
        "E34": 1.0,
        "GFD": 1.5,
        "SA": 1.2,
        "F0": 0.8,
        "F2": 0.8
    }
    
    for fold_idx, test_samples in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx+1}/{n_folds}")
        print(f"Échantillons de test : {test_samples}")
        print('='*50)
        
        data_split = separer_train_test_par_echantillon(data, test_samples, all_methods=True)
        
        first_method = list(data_split.keys())[0]
        n_test = len(data_split[first_method]['X_test'])
        y_test = data_split[first_method]['y_test']
        
        print(f"Taille du test : {n_test} échantillons")
        
        # Évaluer chaque méthode individuellement
        print("\nPerformances individuelles:")
        resultats_ind = evaluer_methodes_individuelles(data_split, k)
        for method, res in resultats_ind.items():
            accuracy = res['accuracy']
            resultats_individuels[method].append(accuracy)
            print(f"  {method:3s}: {accuracy*100:.2f}%")
            # Accumuler pour matrice de confusion
            all_y_true_ind[method].extend(res['y_test'])
            all_y_pred_ind[method].extend(res['y_pred'])
        
        voteur = VoteMajoritaire(k=k, weights=poids_personnalise, use_confidence=True)
        voteur.entrainer(data_split, verbose=False)
        
        print("\nPerformances du vote majoritaire:")
        for mode in modes:
            y_pred, details = voteur.predire_ensemble(data_split, mode=mode)
            accuracy = np.sum(y_pred == y_test) / len(y_test) if len(y_test) > 0 else 0
            resultats_par_mode[mode].append(accuracy)
            print(f"  {mode:20s}: {accuracy*100:.2f}%")
            
            # Accumuler pour matrice de confusion
            all_y_true[mode].extend(y_test)
            all_y_pred[mode].extend(y_pred)
    
    # Résumé final
    print("\nRÉSUMÉ DE LA VALIDATION CROISÉE")
    
    print("\nMoyennes des précisions individuelles:")
    for method in methods:
        if resultats_individuels[method]:
            mean_acc = np.mean(resultats_individuels[method])
            std_acc = np.std(resultats_individuels[method])
            print(f"  {method:3s}: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")
    
    print("\nMoyennes du vote majoritaire:")
    for mode in modes:
        mean_acc = np.mean(resultats_par_mode[mode])
        std_acc = np.std(resultats_par_mode[mode])
        print(f"  {mode:20s}: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")
    
    # Identifier la meilleure méthode
    best_individual = max(
        [(m, np.mean(acc)) for m, acc in resultats_individuels.items() if acc],
        key=lambda x: x[1]
    )
    best_vote = max(
        [(m, np.mean(acc)) for m, acc in resultats_par_mode.items()],
        key=lambda x: x[1]
    )
    
    print("\n" + "-"*50)
    print(f"Meilleure méthode individuelle: {best_individual[0]} ({best_individual[1]*100:.2f}%)")
    print(f"Meilleur mode de vote: {best_vote[0]} ({best_vote[1]*100:.2f}%)")
    
    improvement = (best_vote[1] - best_individual[1]) / best_individual[1] * 100
    if improvement > 0:
        print(f"Le vote majoritaire améliore de {improvement:.1f}%")
    else:
        print(f"Le vote majoritaire dégrade de {abs(improvement):.1f}%")
    
    return {
        'resultats_par_mode': resultats_par_mode,
        'resultats_individuels': resultats_individuels,
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'all_y_true_ind': all_y_true_ind,
        'all_y_pred_ind': all_y_pred_ind,
        'best_mode': best_vote[0],
        'n_classes': n_classes
    }