'''
La methode du vote majoritaire permet d'avoir une vue d'ensemble qui, par prédiction, aura des resultats plus fiables.
Les méthodes font des erreurs différentes. E34 échoue sur les avions, GFD sur les mains. 
En combinant, les forces d'une méthode compensent les faiblesses d'une autre.
'''

"""
Numpy est une bibliotheque que nous allons tres largement utiliser ici.
"""
import numpy as np
from collections import Counter
from utils import (
    charger_donnees,
    predire_classe_avec_confiance,
    separer_train_test_par_echantillon,
    matrice_confusion,
    afficher_matrice_confusion,
    analyser_precision_par_classe
)

# Fixer la seed pour la reproductibilité
np.random.seed(42)

"""
Chemin à modifier en fonction de l'emplacement de la BDD.
"""
base_folder = "C:/Users/lakaf/OneDrive/Bureau/25-26/RF/Projet_RF/data"

# Charger toutes les données
data = charger_donnees(base_folder)

# Liste des méthodes pour référence
methods = ["E34", "GFD", "SA", "F0", "F2"]

# ============================
# Implémentation du Vote Majoritaire
# ============================

class VoteMajoritaire:
    """
    Classifieur par vote majoritaire combinant plusieurs méthodes
    """
    
    def __init__(self, k=5, weights=None, use_confidence=True):
        """
        Paramètres:
        - k: nombre de voisins pour chaque classifieur KNN
        - weights: poids pour chaque méthode (dict ou None pour poids égaux)
        - use_confidence: utiliser la confiance comme poids dans le vote
        """
        self.k = k
        self.weights = weights if weights else {m: 1.0 for m in methods}
        self.use_confidence = use_confidence
        self.data_train = {}
        
    def entrainer(self, data_split):
        """
        Stocke les données d'entraînement pour chaque méthode
        """
        self.data_train = {}
        for method in methods:
            if method in data_split:
                self.data_train[method] = {
                    'X': data_split[method]['X_train'],
                    'y': data_split[method]['y_train']
                }
        print(f"Entraînement terminé avec {len(self.data_train)} méthodes")
        
    def predire_ensemble(self, data_split, mode='majoritaire_simple'):
        """
        Prédit les classes pour l'ensemble de test
        
        Modes disponibles:
        - 'majoritaire_simple': vote simple sans pondération
        - 'majoritaire_pondere': vote pondéré par les poids fixes
        - 'majoritaire_confiance': vote pondéré par la confiance
        - 'moyenne_confiance': moyenne pondérée des confiances
        """
        # Récupérer les échantillons de test (on suppose qu'ils sont les mêmes pour toutes les méthodes)
        first_method = list(data_split.keys())[0]
        n_test = len(data_split[first_method]['X_test'])
        y_test = data_split[first_method]['y_test']
        
        predictions_finales = []
        details_votes = []
        
        for i in range(n_test):
            votes = {}
            confiances = {}
            predictions_par_methode = {}
            
            # Collecter les prédictions de chaque méthode
            for method in methods:
                if method in self.data_train and len(self.data_train[method]['X']) > 0:
                    x_test = data_split[method]['X_test'][i]
                    
                    # Prédire avec confiance
                    classe, confiance = predire_classe_avec_confiance(
                        self.data_train[method]['X'],
                        self.data_train[method]['y'],
                        x_test,
                        self.k
                    )
                    
                    predictions_par_methode[method] = classe
                    
                    if mode == 'majoritaire_simple':
                        # Vote simple: chaque méthode a un poids de 1
                        if classe not in votes:
                            votes[classe] = 0
                        votes[classe] += 1
                        
                    elif mode == 'majoritaire_pondere':
                        # Vote pondéré par les poids fixes
                        if classe not in votes:
                            votes[classe] = 0
                        votes[classe] += self.weights[method]
                        
                    elif mode == 'majoritaire_confiance':
                        # Vote pondéré par la confiance
                        if classe not in votes:
                            votes[classe] = 0
                        if self.use_confidence:
                            votes[classe] += confiance
                        else:
                            votes[classe] += 1
                            
                    elif mode == 'moyenne_confiance':
                        # Moyenne pondérée des confiances
                        if classe not in confiances:
                            confiances[classe] = []
                        confiances[classe].append((confiance, self.weights[method]))
            
            # Déterminer la classe finale
            if mode == 'moyenne_confiance' and confiances:
                # Calculer la moyenne pondérée des confiances pour chaque classe
                moyennes = {}
                for classe, conf_list in confiances.items():
                    total_conf = sum(c * w for c, w in conf_list)
                    total_weight = sum(w for _, w in conf_list)
                    moyennes[classe] = total_conf / total_weight if total_weight > 0 else 0
                classe_finale = max(moyennes, key=moyennes.get)
            elif votes:
                # Vote majoritaire (simple, pondéré ou par confiance)
                classe_finale = max(votes, key=votes.get)
            else:
                # Fallback: classe aléatoire si aucune prédiction
                classe_finale = np.random.randint(1, 10)
            
            predictions_finales.append(classe_finale)
            details_votes.append({
                'predictions': predictions_par_methode,
                'votes': votes,
                'vraie_classe': y_test[i],
                'classe_predite': classe_finale
            })
        
        return np.array(predictions_finales), details_votes

def evaluer_methodes_individuelles(data_split, k=5):
    """
    Évalue chaque méthode individuellement pour comparaison
    """
    resultats = {}
    
    for method in methods:
        if method not in data_split or len(data_split[method]['X_train']) == 0:
            continue
            
        X_train = data_split[method]['X_train']
        y_train = data_split[method]['y_train']
        X_test = data_split[method]['X_test']
        y_test = data_split[method]['y_test']
        
        # Prédire pour chaque échantillon de test
        y_pred = []
        for x in X_test:
            pred, _ = predire_classe_avec_confiance(X_train, y_train, x, k)
            y_pred.append(pred)
        
        y_pred = np.array(y_pred)
        
        # Calculer la précision
        accuracy = np.sum(y_pred == y_test) / len(y_test) if len(y_test) > 0 else 0
        
        resultats[method] = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    return resultats

# ============================
# Validation croisée pour le vote majoritaire
# ============================

def validation_croisee_vote_majoritaire(data, k=5, n_folds=3):
    """
    Validation croisée pour le système de vote majoritaire
    """
    # Créer les folds
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
    
    # Stocker les résultats pour chaque mode
    modes = ['majoritaire_simple', 'majoritaire_pondere', 'majoritaire_confiance']
    resultats_par_mode = {mode: [] for mode in modes}
    resultats_individuels = {method: [] for method in methods}
    
    for fold_idx, test_samples in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_idx+1}/{n_folds}")
        print(f"Échantillons de test : {test_samples}")
        print('='*50)
        
        # Séparer les données
        data_split = separer_train_test_par_echantillon(data, test_samples, all_methods=True)
        
        # Vérifier qu'on a des données
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
        
        # Créer et entraîner le classifieur par vote
        voteur = VoteMajoritaire(k=k, use_confidence=True)
        voteur.entrainer(data_split)
        
        # Tester différents modes de vote
        print("\nPerformances du vote majoritaire:")
        for mode in modes:
            y_pred, details = voteur.predire_ensemble(data_split, mode=mode)
            accuracy = np.sum(y_pred == y_test) / len(y_test) if len(y_test) > 0 else 0
            resultats_par_mode[mode].append(accuracy)
            print(f"  {mode:20s}: {accuracy*100:.2f}%")
        
        # Analyser quelques exemples de votes
        if fold_idx == 0:  # Seulement pour le premier fold
            print("\nExemples de votes (5 premiers):")
            for i in range(min(5, len(details))):
                detail = details[i]
                print(f"\n  Échantillon {i+1}:")
                print(f"    Vraie classe: {detail['vraie_classe']}")
                print(f"    Prédictions par méthode: {detail['predictions']}")
                print(f"    Classe finale: {detail['classe_predite']}")
                if detail['classe_predite'] != detail['vraie_classe']:
                    print(" ERREUR")
    
    # Résumé final
    print("\n" + "="*50)
    print("RÉSUMÉ DE LA VALIDATION CROISÉE")
    print("="*50)
    
    print("\nMoyennes des précisions individuelles:")
    for method in methods:
        if method in resultats_individuels and resultats_individuels[method]:
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
    
    return resultats_par_mode, resultats_individuels

# ============================
# Tests principaux
# ============================

print("="*50)
print("SYSTÈME DE VOTE MAJORITAIRE")
print("="*50)

# Test avec différentes valeurs de k
for k in [3, 5, 7]:
    print(f"\n\n{'#'*50}")
    print(f"TEST AVEC k={k} VOISINS")
    print('#'*50)
    
    resultats_modes, resultats_ind = validation_croisee_vote_majoritaire(
        data, k=k, n_folds=3
    )

# ============================
# Analyse de l'importance des méthodes
# ============================

print("\n\n" + "="*50)
print("ANALYSE DE L'IMPORTANCE DES MÉTHODES")
print("="*50)

# Test en excluant chaque méthode tour à tour
print("\nImpact de l'exclusion de chaque méthode:")
print("(Différence avec le vote complet)")

# D'abord, obtenir la baseline avec toutes les méthodes
echantillons_test = [2, 5, 8]
data_split = separer_train_test_par_echantillon(data, echantillons_test, all_methods=True)

voteur_complet = VoteMajoritaire(k=5)
voteur_complet.entrainer(data_split)
y_pred_complet, _ = voteur_complet.predire_ensemble(data_split, mode='majoritaire_confiance')
y_test = data_split[methods[0]]['y_test']
accuracy_complet = np.sum(y_pred_complet == y_test) / len(y_test)

print(f"Baseline (toutes méthodes): {accuracy_complet*100:.2f}%")

# Tester en excluant chaque méthode
for method_to_exclude in methods:
    # Créer une copie des données sans cette méthode
    methods_subset = [m for m in methods if m != method_to_exclude]
    
    # Reconstruire data_split sans la méthode exclue
    data_split_subset = {m: data_split[m] for m in methods_subset}
    
    # Entraîner et tester
    voteur_subset = VoteMajoritaire(k=5)
    voteur_subset.entrainer(data_split_subset)
    y_pred_subset, _ = voteur_subset.predire_ensemble(data_split_subset, mode='majoritaire_confiance')
    
    accuracy_subset = np.sum(y_pred_subset == y_test) / len(y_test)
    
    difference = accuracy_complet - accuracy_subset
    if difference > 0:
        print(f"  Sans {method_to_exclude:3s}: {accuracy_subset*100:.2f}% (perte: -{difference*100:.2f}%)")
    else:
        print(f"  Sans {method_to_exclude:3s}: {accuracy_subset*100:.2f}% (gain: +{abs(difference)*100:.2f}%)")

# ============================
# Matrice de confusion pour le meilleur système
# ============================

print("\n\n" + "="*50)
print("MATRICE DE CONFUSION DU MEILLEUR SYSTÈME")
print("="*50)

# Utiliser le meilleur k trouvé précédemment
best_k = 5
data_split = separer_train_test_par_echantillon(data, [2, 5, 8], all_methods=True)

voteur_final = VoteMajoritaire(k=best_k)
voteur_final.entrainer(data_split)

# Prédire avec le meilleur mode
y_pred_final, details_final = voteur_final.predire_ensemble(
    data_split, mode='majoritaire_confiance'
)
y_test_final = data_split[methods[0]]['y_test']

# Créer la matrice de confusion
confusion = matrice_confusion(y_test_final, y_pred_final)

print("\nMatrice de confusion (lignes=vraies, colonnes=prédites):")
afficher_matrice_confusion(confusion)

# Précision par classe
print("\nPrécision par classe:")
analyser_precision_par_classe(y_test_final, y_pred_final)

# ============================
# Analyse des désaccords entre méthodes
# ============================

print("\n\n" + "="*50)
print("ANALYSE DES DÉSACCORDS ENTRE MÉTHODES")
print("="*50)

# Compter les cas où les méthodes ne sont pas d'accord
accords_complets = 0
desaccords_totaux = 0
desaccords_partiels = {}

for detail in details_final:
    predictions = list(detail['predictions'].values())
    unique_predictions = set(predictions)
    
    if len(unique_predictions) == 1:
        accords_complets += 1
    else:
        desaccords_totaux += 1
        nb_classes_differentes = len(unique_predictions)
        if nb_classes_differentes not in desaccords_partiels:
            desaccords_partiels[nb_classes_differentes] = 0
        desaccords_partiels[nb_classes_differentes] += 1

print(f"Accord complet (toutes méthodes): {accords_complets}/{len(details_final)} ({accords_complets/len(details_final)*100:.1f}%)")
print(f"Désaccord (au moins 2 méthodes différentes): {desaccords_totaux}/{len(details_final)} ({desaccords_totaux/len(details_final)*100:.1f}%)")

if desaccords_partiels:
    print("\nDétail des désaccords:")
    for nb_classes, count in sorted(desaccords_partiels.items()):
        print(f"  {nb_classes} classes différentes prédites: {count} cas")

# Analyser la corrélation avec les erreurs
print("\nRelation désaccord/erreur:")
erreurs_si_accord = 0
erreurs_si_desaccord = 0
total_accord = 0
total_desaccord = 0

for i, detail in enumerate(details_final):
    predictions = list(detail['predictions'].values())
    unique_predictions = set(predictions)
    est_erreur = (y_pred_final[i] != y_test_final[i])
    
    if len(unique_predictions) == 1:
        total_accord += 1
        if est_erreur:
            erreurs_si_accord += 1
    else:
        total_desaccord += 1
        if est_erreur:
            erreurs_si_desaccord += 1

if total_accord > 0:
    taux_erreur_accord = erreurs_si_accord / total_accord * 100
    print(f"  Taux d'erreur quand accord: {taux_erreur_accord:.1f}%")

if total_desaccord > 0:
    taux_erreur_desaccord = erreurs_si_desaccord / total_desaccord * 100
    print(f"  Taux d'erreur quand désaccord: {taux_erreur_desaccord:.1f}%")
    print(f"  → Le désaccord est un bon indicateur d'incertitude!")

print("\n" + "="*50)
print("FIN DE L'ANALYSE DU VOTE MAJORITAIRE")
print("="*50)
