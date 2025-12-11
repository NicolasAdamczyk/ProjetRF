# import des modules requis
import numpy as np # librairie pour les operations numeriques
# import de la fct de separation, suppose etre dans utils.py
from utils import separer_train_test_par_echantillon 
# import de la fct de prediction knn avec confiance, suppose etre dans knn.py
from knn import predire_classe_avec_confiance 

# liste des methodes (descripteurs) a evaluer/combiner
methods = ["e34", "gfd", "sa", "f0", "f2"] 

class VoteMajoritaire:
    """
    classifieur par vote majoritaire combinant plusieurs methodes knn.
    

    args:
        k (int): nb de voisins pour knn.
        weights (dict): poids de chaque methode. si none, poids = 1.0 pour toutes.
        use_confidence (bool): si true, pondere le vote par la confiance des knn. (n'est plus utilise directement, la confiance est integree dans 'majoritaire_confiance')
    """

    def __init__(self, k=5, weights=None, use_confidence=True):
        """
        constructeur initialisant le classifieur votemajoritaire.
        """
        self.k = k # nb de voisins pour les knn sous-jacents
        self.use_confidence = use_confidence # flag d'utilisation de la confiance (non directement utilise pour le moment)
        # init des poids : utilises si le mode 'majoritaire_pondere' ou 'majoritaire_confiance' est choisi
        self.weights = weights if weights else {m: 1.0 for m in methods} 
        self.data_train = {} # stocke x_train et y_train par methode

    def entrainer(self, data_split, verbose=True):
        """
        stocke les donnees d'entrainement pour chaque methode disponible. (etape d'entrainement du classifieur d'ensemble)

        args:
            data_split (dict): donnees separees train/test pour chaque methode.
            verbose (bool): si true, affiche le nb de methodes entrainees.
        """
        self.data_train = {} # reinit des donnees d'entrainement
        for method in methods:
            if method in data_split:
                # stocke les donnees d'entrainement de la methode
                self.data_train[method] = {
                    'x': data_split[method]['x_train'],
                    'y': data_split[method]['y_train']
                }
        if verbose:
            print(f"entrainement termine avec {len(self.data_train)} methodes")

    def predire_ensemble(self, data_split, mode="majoritaire_simple"):
        """
        predit les classes pour l'ensemble test en combinant les votes des differentes methodes.

        args:
            data_split (dict): donnees separees train/test (utilise x_test).
            mode (str): 3 strategies de vote:
                - "majoritaire_simple": un vote par methode
                - "majoritaire_pondere": vote pondere par poids predefinis (self.weights)
                - "majoritaire_confiance": vote pondere par confiance knn et poids
        returns:
            tuple: (np.ndarray: classes predites finales, list: details des votes pour chaque echantillon)
        """
        # recupere le nb d'echantillons test (en utilisant la premiere methode disponible comme ref)
        first_method = list(data_split.keys())[0]
        n_test = len(data_split[first_method]['x_test'])
        y_test = data_split[first_method]['y_test'] # vrais labels test
        
        predictions_finales = []
        details_votes = []
        
        # boucle sur chaque echantillon de test
        for i in range(n_test):
            votes = {} # dict pour accumuler les votes (classe -> score/poids)
            predictions_par_methode = {} # dict pour stocker les predictions de chaque methode
            
            # boucle sur chaque methode
            for method in methods:
                # verif si la methode est entrainee et si x_train n'est pas vide
                if method in self.data_train and len(self.data_train[method]['x']) > 0:
                    x_test = data_split[method]['x_test'][i] # vecteur test de la methode
                    
                    # prediction knn avec score de confiance
                    classe, confiance = predire_classe_avec_confiance(
                        self.data_train[method]['x'],
                        self.data_train[method]['y'],
                        x_test,
                        self.k
                    )
                    
                    predictions_par_methode[method] = classe
                    
                    if classe not in votes:
                        votes[classe] = 0.0 # init vote avec 0.0 pour les scores flotants
                    
                    # accumulation des votes selon le mode choisi
                    if mode == 'majoritaire_simple':
                        votes[classe] += 1 # 1 vote par methode
                    elif mode == 'majoritaire_pondere':
                        votes[classe] += self.weights[method] # vote = poids predefini
                    elif mode == 'majoritaire_confiance':
                        # vote = confiance knn * poids predefini
                        votes[classe] += confiance * self.weights[method] 

            # decision finale: classe avec le maximum de votes/scores
            if votes:
                classe_finale = max(votes, key=votes.get) # cle (classe) avec la valeur (score) max
            else:
                classe_finale = 1 # valeur par defaut si aucun vote
            
            predictions_finales.append(classe_finale)
            details_votes.append({
                'predictions': predictions_par_methode,
                'votes': votes,
                'vraie_classe': y_test[i],
                'classe_predite': classe_finale
            })
        
        return np.array(predictions_finales), details_votes


# ============================================================
# evaluation des descripteurs individuels
# ============================================================

def evaluer_methodes_individuelles(data_split, k=5):
    """
    evalue chaque methode individuellement avec knn.

    args:
        data_split (dict): donnees separees train/test pour chaque methode.
        k (int): nb de voisins knn.

    returns:
        dict: resultats par methode avec 'accuracy', 'y_pred' et 'y_test'.
    """
    resultats = {}
    
    for method in methods:
        # verif que la methode existe et que les donnees train ne sont pas vides
        if method not in data_split or len(data_split[method]['x_train']) == 0:
            continue
            
        x_train = data_split[method]['x_train']
        y_train = data_split[method]['y_train']
        x_test = data_split[method]['x_test']
        y_test = data_split[method]['y_test']
        
        y_pred = []
        # prediction de chaque echantillon test
        for x in x_test:
            # on utilise la fct de prediction avec confiance, mais on ne garde que la prediction (pred)
            pred, _ = predire_classe_avec_confiance(x_train, y_train, x, k)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        # calcul de l'accuracy
        accuracy = np.sum(y_pred == y_test) / len(y_test) if len(y_test) > 0 else 0
        
        resultats[method] = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    return resultats

def filtrer_donnees_classes(data, classes_a_garder):
    """
    filtre les donnees pour ne garder que certaines classes.

    args:
        data (dict): dict de donnees original.
        classes_a_garder (list): classes a conserver.

    returns:
        dict: dict filtre.
    """
    data_filtre = {}
    # parcours de tous les echantillons
    for key, value in data.items():
        # ne garde que les echantillons dont la classe est dans la liste
        if value["class"] in classes_a_garder:
            data_filtre[key] = value
    return data_filtre

def validation_croisee_complete(data, k=5, n_folds=3, n_classes=9):
    """
    effectue une validation croisee complete combinant toutes les methodes et tous les modes de vote.

    args:
        data (dict): donnees originales.
        k (int): nb de voisins knn.
        n_folds (int): nb de plis (folds) pour la validation croisee.
        n_classes (int): nb de classes dans le dataset.

    returns:
        dict: dict contenant les resultats agreges.
    """
    # preparation des folds pour la validation croisee par echantillon
    echantillons = list(range(1, 12)) # suppose 11 echantillons (1 a 11)
    np.random.shuffle(echantillons) # melange des echantillons
    
    fold_size = len(echantillons) // n_folds
    folds = []
    # creation des listes d'echantillons pour chaque fold
    for i in range(n_folds):
        start = i * fold_size
        if i == n_folds - 1:
            folds.append(echantillons[start:]) # le dernier fold prend le reste
        else:
            folds.append(echantillons[start:start + fold_size])
    
    modes = ['majoritaire_simple', 'majoritaire_pondere', 'majoritaire_confiance']
    resultats_par_mode = {mode: [] for mode in modes} # stockage des accuracies par mode
    resultats_individuels = {method: [] for method in methods} # stockage des accuracies par methode
    
    # accumulation pour matrice de confusion (sur tous les folds)
    all_y_true = {mode: [] for mode in modes}
    all_y_pred = {mode: [] for mode in modes}
    all_y_true_ind = {m: [] for m in methods}
    all_y_pred_ind = {m: [] for m in methods}
    
    # definition des poids personnalises pour le mode pondere
    poids_personnalise = {
        "e34": 1.0,
        "gfd": 1.5, # donne plus de poids a gfd
        "sa": 1.2,
        "f0": 0.8,
        "f2": 0.8
    }
    
    # boucle sur chaque fold
    for fold_idx, test_samples in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"fold {fold_idx+1}/{n_folds}")
        print(f"echantillons de test : {test_samples}")
        print('='*50)
        
        # separation train/test des donnees (all_methods=true)
        data_split = separer_train_test_par_echantillon(data, test_samples, all_methods=True)
        
        first_method = list(data_split.keys())[0]
        n_test = len(data_split[first_method]['x_test'])
        y_test = data_split[first_method]['y_test']
        
        print(f"taille du test : {n_test} echantillons")
        
        # evaluer chaque methode individuellement
        print("\nperformances individuelles:")
        resultats_ind = evaluer_methodes_individuelles(data_split, k)
        for method, res in resultats_ind.items():
            accuracy = res['accuracy']
            resultats_individuels[method].append(accuracy)
            print(f"  {method:3s}: {accuracy*100:.2f}%")
            # accumuler pour matrice de confusion
            all_y_true_ind[method].extend(res['y_test'])
            all_y_pred_ind[method].extend(res['y_pred'])
            
        # initialisation et entrainement du vote majoritaire
        voteur = VoteMajoritaire(k=k, weights=poids_personnalise, use_confidence=True)
        voteur.entrainer(data_split, verbose=False)
        
        # evaluation des modes de vote
        print("\nperformances du vote majoritaire:")
        for mode in modes:
            y_pred, details = voteur.predire_ensemble(data_split, mode=mode)
            accuracy = np.sum(y_pred == y_test) / len(y_test) if len(y_test) > 0 else 0
            resultats_par_mode[mode].append(accuracy)
            print(f"  {mode:20s}: {accuracy*100:.2f}%")
            
            # accumuler pour matrice de confusion
            all_y_true[mode].extend(y_test)
            all_y_pred[mode].extend(y_pred)
    
    # resume final
    print("\nresume de la validation croisee")
    
    print("\nmoyennes des precisions individuelles:")
    for method in methods:
        if resultats_individuels[method]:
            mean_acc = np.mean(resultats_individuels[method])
            std_acc = np.std(resultats_individuels[method])
            print(f"  {method:3s}: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")
    
    print("\nmoyennes du vote majoritaire:")
    for mode in modes:
        mean_acc = np.mean(resultats_par_mode[mode])
        std_acc = np.std(resultats_par_mode[mode])
        print(f"  {mode:20s}: {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")
    
    # identifier la meilleure methode (individuelle et vote)
    best_individual = max(
        [(m, np.mean(acc)) for m, acc in resultats_individuels.items() if acc],
        key=lambda x: x[1]
    )
    best_vote = max(
        [(m, np.mean(acc)) for m, acc in resultats_par_mode.items()],
        key=lambda x: x[1]
    )
    
    print("\n" + "-"*50)
    print(f"meilleure methode individuelle: {best_individual[0]} ({best_individual[1]*100:.2f}%)")
    print(f"meilleur mode de vote: {best_vote[0]} ({best_vote[1]*100:.2f}%)")
    
    # calcul de l'amelioration
    improvement = (best_vote[1] - best_individual[1]) / best_individual[1] * 100
    if improvement > 0:
        print(f"le vote majoritaire ameliore de {improvement:.1f}%")
    else:
        print(f"le vote majoritaire degrade de {abs(improvement):.1f}%")
    
    # retour des resultats
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
