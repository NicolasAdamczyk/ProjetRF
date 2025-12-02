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

# Chemin à modifier selon l'emplacement des données
base_folder = "C:/Users/lakaf/OneDrive/Bureau/25-26/RF/Projet_RF/data"

# Charger toutes les données
data = charger_donnees(base_folder)

# Liste des méthodes
methods = ["E34", "GFD", "SA", "F0", "F2"]


class VoteMajoritaire:
    """
    Classifieur par vote majoritaire combinant plusieurs méthodes KNN.
    """
    
    def __init__(self, k=5, weights=None, use_confidence=True):
        self.k = k
        self.weights = weights if weights else {m: 1.0 for m in methods}
        self.use_confidence = use_confidence
        self.data_train = {}
        
    def entrainer(self, data_split, verbose=True):
        self.data_train = {}
        for method in methods:
            if method in data_split:
                self.data_train[method] = {
                    'X': data_split[method]['X_train'],
                    'y': data_split[method]['y_train']
                }
        if verbose:
            print(f"Entraînement terminé avec {len(self.data_train)} méthodes")
        
    def predire_ensemble(self, data_split, mode='majoritaire_simple'):
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
                    
                    classe, confiance = predire_classe_avec_confiance(
                        self.data_train[method]['X'],
                        self.data_train[method]['y'],
                        x_test,
                        self.k
                    )
                    
                    predictions_par_methode[method] = classe
                    
                    if classe not in votes:
                        votes[classe] = 0
                    
                    if mode == 'majoritaire_simple':
                        votes[classe] += 1
                    elif mode == 'majoritaire_pondere':
                        votes[classe] += self.weights[method]
                    elif mode == 'majoritaire_confiance':
                        votes[classe] += confiance * self.weights[method]
            
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


def evaluer_methodes_individuelles(data_split, k=5):
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
    """
    data_filtre = {}
    for key, value in data.items():
        if value["class"] in classes_a_garder:
            data_filtre[key] = value
    return data_filtre


def validation_croisee_complete(data, k=5, n_folds=3, n_classes=9):
    """
    Validation croisée qui accumule TOUTES les prédictions pour une matrice
    de confusion représentative.
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
    
    # Résumé
    print("\n" + "="*50)
    print("RÉSUMÉ DE LA VALIDATION CROISÉE")
    print("="*50)
    
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


def calculer_metriques_detaillees(y_true, y_pred, n_classes=9):
    """
    Calcule précision, rappel, F-mesure pour chaque classe.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metriques = {}
    
    for c in range(1, n_classes + 1):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        rappel = tp / (tp + fn) if (tp + fn) > 0 else 0
        f_mesure = 2 * precision * rappel / (precision + rappel) if (precision + rappel) > 0 else 0
        
        metriques[c] = {
            'precision': precision,
            'rappel': rappel,
            'f_mesure': f_mesure,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Moyennes macro
    metriques['macro'] = {
        'precision': np.mean([m['precision'] for m in metriques.values() if isinstance(m, dict) and 'precision' in m]),
        'rappel': np.mean([m['rappel'] for m in metriques.values() if isinstance(m, dict) and 'rappel' in m]),
        'f_mesure': np.mean([m['f_mesure'] for m in metriques.values() if isinstance(m, dict) and 'f_mesure' in m])
    }
    
    return metriques


def afficher_metriques(metriques, n_classes=9):
    """Affiche les métriques de manière formatée."""
    print("\n{:^8} {:^12} {:^12} {:^12}".format("Classe", "Précision", "Rappel", "F-mesure"))
    print("-" * 48)
    
    for c in range(1, n_classes + 1):
        if c in metriques:
            m = metriques[c]
            print(f"{c:^8} {m['precision']*100:^12.2f} {m['rappel']*100:^12.2f} {m['f_mesure']*100:^12.2f}")
    
    print("-" * 48)
    macro = metriques['macro']
    print(f"{'Macro':^8} {macro['precision']*100:^12.2f} {macro['rappel']*100:^12.2f} {macro['f_mesure']*100:^12.2f}")


# ============================
# TESTS PRINCIPAUX - 9 CLASSES
# ============================

print("="*60)
print("SYSTÈME DE VOTE MAJORITAIRE - ANALYSE COMPLÈTE")
print("="*60)

# Test avec k=3 (meilleur selon les résultats précédents)
print("\n\n" + "#"*60)
print("TEST AVEC k=3 VOISINS (9 CLASSES)")
print("#"*60)

resultats_9classes = validation_croisee_complete(data, k=3, n_folds=3, n_classes=9)

# Matrice de confusion du meilleur mode sur TOUTE la validation croisée
best_mode = resultats_9classes['best_mode']
y_true_all = np.array(resultats_9classes['all_y_true'][best_mode])
y_pred_all = np.array(resultats_9classes['all_y_pred'][best_mode])

print("\n\n" + "="*60)
print(f"MATRICE DE CONFUSION - {best_mode.upper()} (9 CLASSES)")
print("(Agrégée sur tous les folds de validation croisée)")
print("="*60)

confusion_9 = matrice_confusion(y_true_all, y_pred_all, n_classes=9)
print("\nMatrice de confusion (lignes=vraies, colonnes=prédites):")
afficher_matrice_confusion(confusion_9)

print("\nMétriques détaillées par classe:")
metriques_9 = calculer_metriques_detaillees(y_true_all, y_pred_all, n_classes=9)
afficher_metriques(metriques_9, n_classes=9)

# Matrice pour GFD seul
print("\n\n" + "="*60)
print("MATRICE DE CONFUSION - GFD SEUL (9 CLASSES)")
print("="*60)

y_true_gfd = np.array(resultats_9classes['all_y_true_ind']['GFD'])
y_pred_gfd = np.array(resultats_9classes['all_y_pred_ind']['GFD'])

confusion_gfd = matrice_confusion(y_true_gfd, y_pred_gfd, n_classes=9)
print("\nMatrice de confusion:")
afficher_matrice_confusion(confusion_gfd)

print("\nMétriques détaillées:")
metriques_gfd = calculer_metriques_detaillees(y_true_gfd, y_pred_gfd, n_classes=9)
afficher_metriques(metriques_gfd, n_classes=9)


# ============================
# TESTS - 5 PREMIÈRES CLASSES
# ============================

print("\n\n" + "#"*60)
print("COMPARAISON AVEC 5 PREMIÈRES CLASSES UNIQUEMENT")
print("#"*60)

# Filtrer les données pour les 5 premières classes
data_5classes = filtrer_donnees_classes(data, [1, 2, 3, 4, 5])
print(f"\nNombre d'images après filtrage: {len(data_5classes)}")

# Reset seed pour reproductibilité
np.random.seed(42)

resultats_5classes = validation_croisee_complete(data_5classes, k=3, n_folds=3, n_classes=5)

# Matrice de confusion pour 5 classes
best_mode_5 = resultats_5classes['best_mode']
y_true_5 = np.array(resultats_5classes['all_y_true'][best_mode_5])
y_pred_5 = np.array(resultats_5classes['all_y_pred'][best_mode_5])

print("\n\n" + "="*60)
print(f"MATRICE DE CONFUSION - {best_mode_5.upper()} (5 CLASSES)")
print("="*60)

confusion_5 = matrice_confusion(y_true_5, y_pred_5, n_classes=5)
print("\nMatrice de confusion (lignes=vraies, colonnes=prédites):")
# Affichage adapté pour 5 classes
print("    ", " ".join([f"c{i}" for i in range(1, 6)]))
for i in range(5):
    print(f"c{i+1}  ", " ".join([f"{confusion_5[i,j]:2d}" for j in range(5)]))

print("\nMétriques détaillées par classe:")
metriques_5 = calculer_metriques_detaillees(y_true_5, y_pred_5, n_classes=5)
afficher_metriques(metriques_5, n_classes=5)

# GFD seul sur 5 classes
print("\n" + "="*60)
print("GFD SEUL (5 CLASSES)")
print("="*60)

y_true_gfd5 = np.array(resultats_5classes['all_y_true_ind']['GFD'])
y_pred_gfd5 = np.array(resultats_5classes['all_y_pred_ind']['GFD'])

metriques_gfd5 = calculer_metriques_detaillees(y_true_gfd5, y_pred_gfd5, n_classes=5)
afficher_metriques(metriques_gfd5, n_classes=5)


# ============================
# TABLEAU COMPARATIF FINAL
# ============================

print("\n\n" + "="*60)
print("TABLEAU COMPARATIF FINAL")
print("="*60)

print("\n{:<25} {:>15} {:>15}".format("Méthode/Mode", "9 classes", "5 classes"))
print("-" * 55)

# Méthodes individuelles
for method in methods:
    acc_9 = np.mean(resultats_9classes['resultats_individuels'][method]) * 100
    acc_5 = np.mean(resultats_5classes['resultats_individuels'][method]) * 100
    print(f"{method:<25} {acc_9:>14.2f}% {acc_5:>14.2f}%")

print("-" * 55)

# Vote majoritaire
for mode in ['majoritaire_simple', 'majoritaire_pondere', 'majoritaire_confiance']:
    acc_9 = np.mean(resultats_9classes['resultats_par_mode'][mode]) * 100
    acc_5 = np.mean(resultats_5classes['resultats_par_mode'][mode]) * 100
    print(f"{mode:<25} {acc_9:>14.2f}% {acc_5:>14.2f}%")


# ============================
# ANALYSE DES CLASSES PROBLÉMATIQUES
# ============================

print("\n\n" + "="*60)
print("ANALYSE DES CLASSES PROBLÉMATIQUES")
print("="*60)

print("\nClasses difficiles (rappel < 90%) avec vote majoritaire (9 classes):")
for c in range(1, 10):
    if c in metriques_9 and metriques_9[c]['rappel'] < 0.9:
        m = metriques_9[c]
        print(f"  Classe {c}: Rappel={m['rappel']*100:.1f}%, Précision={m['precision']*100:.1f}%")

print("\nComparaison rappel GFD vs Vote (9 classes):")
for c in range(1, 10):
    rappel_gfd = metriques_gfd[c]['rappel'] if c in metriques_gfd else 0
    rappel_vote = metriques_9[c]['rappel'] if c in metriques_9 else 0
    diff = (rappel_vote - rappel_gfd) * 100
    symbole = "+" if diff >= 0 else ""
    print(f"  Classe {c}: GFD={rappel_gfd*100:.1f}% → Vote={rappel_vote*100:.1f}% ({symbole}{diff:.1f}%)")


# ============================
# ANALYSE DE L'IMPORTANCE DES MÉTHODES
# ============================

print("\n\n" + "="*60)
print("ANALYSE DE L'IMPORTANCE DES MÉTHODES")
print("="*60)

np.random.seed(42)
echantillons_test = [2, 5, 8]
data_split = separer_train_test_par_echantillon(data, echantillons_test, all_methods=True)

voteur_complet = VoteMajoritaire(k=3)
voteur_complet.entrainer(data_split, verbose=False)
y_pred_complet, _ = voteur_complet.predire_ensemble(data_split, mode='majoritaire_confiance')
y_test = data_split[methods[0]]['y_test']
accuracy_complet = np.sum(y_pred_complet == y_test) / len(y_test)

print(f"\nBaseline (toutes méthodes): {accuracy_complet*100:.2f}%")
print("\nImpact de l'exclusion de chaque méthode:")

for method_to_exclude in methods:
    methods_subset = [m for m in methods if m != method_to_exclude]
    data_split_subset = {m: data_split[m] for m in methods_subset}
    
    voteur_subset = VoteMajoritaire(k=3)
    voteur_subset.entrainer(data_split_subset, verbose=False)
    y_pred_subset, _ = voteur_subset.predire_ensemble(data_split_subset, mode='majoritaire_confiance')
    
    accuracy_subset = np.sum(y_pred_subset == y_test) / len(y_test)
    difference = (accuracy_complet - accuracy_subset) * 100
    
    print(f"  Sans {method_to_exclude:3s}: {accuracy_subset*100:.2f}% (impact: {difference:+.2f}%)")


print("\n" + "="*60)
print("FIN DE L'ANALYSE")
print("="*60)
