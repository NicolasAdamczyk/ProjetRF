import numpy as np
from data_loader import charger_donnees
from utils import separer_train_test_par_echantillon, afficher_matrice_confusion, afficher_classification_report

from knn import validation_croisee_knn, predire_knn
from kmeans import kmeans, evaluer_clustering, mapper_clusters_vers_classes, validation_croisee_kmeans
from vote_majoritaire import validation_croisee_complete, filtrer_donnees_classes, VoteMajoritaire

# ============================
# CONFIGURATION GLOBALE DES CLASSES ET ECHANTILLONS
# ============================
np.random.seed(42)  # reproductibilité pour la comparaison des résultats
base_folder = "./data"
classes = range(1, 10)  # s01 à s09
samples = range(1, 12)  # n001 à n011
methods = ["E34", "GFD", "SA", "F0", "F2"] # Descripteurs

# ============================
# CHARGEMENT DES DONNÉES
# ============================
data = charger_donnees(base_folder, classes, samples, methods)
print("Lecture terminée.")
print("Nombre total d'images :", len(data))

echantillons_test = [2, 5, 8] # échantillons réservés pour les tests fixes (à modifier pour comparer les résultats)

# ============================
# TEST KNN SUR UN SIMPLE SPLIT FIXE
# ============================
print("\n" + "="*50)
print("TEST KNN SIMPLE SUR SPLIT FIXE")
print("="*50)

for method in methods:
    print(f"\nMéthode : {method}")
    
    # Séparer train/test pour cette méthode
    X_train, y_train, X_test, y_test = separer_train_test_par_echantillon(data, echantillons_test, method) # Utilisation de la fonction utilitaire (voir utils.py)
    print(f"Échantillons de test : {echantillons_test}")
    
    # Prédiction KNN pour chaque instance de test
    y_pred = [predire_knn(X_train, y_train, x, k=5) for x in X_test] # k=5 voisins, (voir knn.py)

    # Calcul et affichage de la précision
    accuracy = np.mean(np.array(y_pred) == np.array(y_test))
    print(f"Accuracy : {accuracy*100:.2f}%\n")
    
    # Affichage de la matrice de confusion (voir utils.py)
    print("Matrice de Confusion :")
    afficher_matrice_confusion(y_test, y_pred, classes, titre=f"KNN - Matrice de Confusion ({method})")
    
    # Affichage du rapport de classification complet (precision, rappel, F1-score), (voir utils.py)
    print("\nRapport de classification :\n")
    afficher_classification_report(y_test, y_pred, classes)


# ============================
# TEST KNN AVEC VALIDATION CROISEE
# ============================
print("\n" + "="*50)
print("TEST KNN AVEC VALIDATION CROISEE")
print("="*50)

for method in methods:
    # Préparer X et y pour toute la méthode
    X = [data[key][method] for key in data if method in data[key]]
    y = [data[key]["class"] for key in data if method in data[key]]

    print(f"\nMéthode : {method}")
    for k in [1, 3, 5, 7]:
        mean_acc, std_acc, accs = validation_croisee_knn(X, y, k=k, n_folds=5) # (voir knn.py), utilisation de 5 folds
        print(f"  k={k}, Moyenne : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  Accuracies par fold : {[f'{a*100:.2f}%' for a in accs]}")

# ============================
# TEST K-MEANS
# ============================
print("\n" + "="*50)
print("TEST K-MEANS")
print("="*50)

kmeans_results = {} # pour stocker éventuellement les résultats par méthode (ne sert à rien actuellement)

for method in methods:
    print(f"\nMéthode : {method}")
    
    # Préparer les données X et les labels y_true
    X = np.array([data[key][method] for key in data if method in data[key]])
    y_true = np.array([data[key]["class"] for key in data if method in data[key]])

    # Nombre de clusters = nombre de classes
    k = 9

    # Appliquer l'algorithme K-Means
    y_pred, centroids, inertia = kmeans(X, k, max_iter=100, init='kmeans++', n_init=10) # (voir kmeans.py)

    # Évaluation clustering : pureté et accuracy
    purity = evaluer_clustering(y_true, y_pred) # (voir kmeans.py)
    y_mapped, mapping = mapper_clusters_vers_classes(y_pred, y_true) # (voir kmeans.py)
    accuracy = np.sum(y_mapped == y_true) / len(y_true)

    print(f"Inertie: {inertia:.2f}, Pureté: {purity*100:.2f}%, Accuracy: {accuracy*100:.2f}%")

    # Affichage de la matrice de confusion (voir utils.py)
    afficher_matrice_confusion(y_true, y_mapped, classes, titre=f"K-Means – Matrice de Confusion ({method})")
    
    # Affichage du rapport de classification complet (precision, rappel, F1-score), (voir utils.py)
    afficher_classification_report(y_true, y_mapped, classes)

# ============================
# VALIDATION CROISÉE K-MEANS
# ============================
print("\n" + "="*50)
print("VALIDATION CROISÉE K-MEANS")
print("="*50)

for method in methods:
    print(f"\nMéthode : {method}")

    X = [data[key][method] for key in data if method in data[key]]
    y = [data[key]["class"] for key in data if method in data[key]]

    mean_acc, std_acc, accs = validation_croisee_kmeans(
        X, y, k=9, n_folds=5, max_iter=200, init="kmeans++", n_init=10
    ) # (voir kmeans.py)

    print(f"  Moyenne : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Accuracies : {[f'{a*100:.2f}%' for a in accs]}")

# ============================
# K-MEANS - METHODE DU COUDE 
# Permet d'estimer le nombre optimal de clusters en observant la diminution de l'inertie.
# ============================
print("\n\n" + "="*50)
print("METHODE DU COUDE")
print("="*50)

method = "GFD"  # Méthode à tester pour la recherche du nombre optimal de clusters (à modifier si besoin)

# Préparer les données
X = np.array([data[key][method] for key in data if method in data[key]])
y_true = np.array([data[key]["class"] for key in data if method in data[key]])

# Tester différentes valeurs de k
k_values = range(2, 16)
inertias = []
purities = []

print(f"\nCalcul pour différentes valeurs de k ({method}):")
for k in k_values:
    y_pred, centroids, inertia = kmeans(X, k, max_iter=100, init='kmeans++', n_init=5)
    purity = evaluer_clustering(y_true, y_pred)
    inertias.append(inertia)
    purities.append(purity)
    print(f"  k={k:2d}: Inertie={inertia:8.2f}, Pureté={purity*100:.2f}%")

# Calculer le ratio d'amélioration de l'inertie
print("\nRatio d'amélioration de l'inertie:")
for i in range(1, len(inertias)):
    improvement = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
    print(f"  k={k_values[i-1]} -> k={k_values[i]}: {improvement:.2f}%")

# Suggestion du nombre optimal de clusters (point où l'amélioration ralentit)
improvements = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
k_optimal = 9
for i in range(1, len(improvements)):
    if improvements[i] < improvements[i-1] * 0.5:  # Si l'amélioration chute de plus de 50%
        k_optimal = k_values[i]
        break

print(f"\nNombre de clusters suggéré: {k_optimal}")
print("(Basé sur la méthode du coude - point où l'amélioration ralentit)")

# ============================================================
# VOTE MAJORITAIRE TEST PRINCIPAL – 9 CLASSES
method_to_test = "SA" # changer ici la méthode à tester pour les tests individuels
# ============================================================

print("\n" + "="*60)
print("VALIDATION CROISEE COMPLETE (VOTE MAJORITAIRE) - 9 CLASSES")
print("="*60)

resultats_9 = validation_croisee_complete(data, k=5, n_folds=3, n_classes=9) # (voir vote_majoritaire.py)

best_mode = resultats_9["best_mode"]
print(f"\n>>> Meilleur mode identifié : {best_mode}\n")

y_true_all = np.array(resultats_9["all_y_true"][best_mode])
y_pred_all = np.array(resultats_9["all_y_pred"][best_mode])

# Affichage de la matrice de confusion (voir utils.py)
print(f"MATRICE DE CONFUSION – {best_mode.upper()} (9 classes)")
afficher_matrice_confusion(y_true_all, y_pred_all, classes)

print("-"*40)

# Affichage du rapport de classification complet (precision, rappel, F1-score), (voir utils.py)
print("\nRAPPORT DE CLASSIFICATION - 9 CLASSES")
afficher_classification_report(y_true_all, y_pred_all, classes)


# ============================================================
# TEST INDIVIDUEL – méthode choisie (9 CLASSES)
# ============================================================
print("\n" + "="*60)
print(f"{method_to_test} SEUL – 9 CLASSES")
print("="*60)

y_true_method9 = np.array(resultats_9["all_y_true_ind"][method_to_test])
y_pred_method9 = np.array(resultats_9["all_y_pred_ind"][method_to_test])

# Affichage de la matrice de confusion (voir utils.py)
afficher_matrice_confusion(y_true_method9, y_pred_method9, classes)

print("\nRapport de classification :")
afficher_classification_report(y_true_method9, y_pred_method9, classes)


# ============================================================
# VALIDATION CROISEE – 5 PREMIÈRES CLASSES
# ============================================================
print("\n" + "="*60)
print("VALIDATION CROISEE COMPLETE – 5 PREMIÈRES CLASSES")
print("="*60)

data5 = filtrer_donnees_classes(data, [1, 2, 3, 4, 5])
print("Nombre d’images après filtrage :", len(data5))

resultats_5 = validation_croisee_complete(data5, k=5, n_folds=3, n_classes=5)

best_mode_5 = resultats_5["best_mode"]

y_true_5 = np.array(resultats_5["all_y_true"][best_mode_5])
y_pred_5 = np.array(resultats_5["all_y_pred"][best_mode_5])

# Affichage de la matrice de confusion (voir utils.py)
print(f"MATRICE DE CONFUSION – {best_mode_5.upper()} (5 classes)")
afficher_matrice_confusion(y_true_5, y_pred_5, classes=list(range(1, 6)))

print("-"*40)

# Affichage du rapport de classification complet (precision, rappel, F1-score), (voir utils.py)
print("\nRAPPORT DE CLASSIFICATION – 5 CLASSES")
afficher_classification_report(y_true_5, y_pred_5, classes=list(range(1, 6)))


# ============================================================
# TEST INDIVIDUEL – méthode choisie (5 CLASSES)
# ============================================================
print("\n" + "="*60)
print(f"{method_to_test} SEUL – 5 CLASSES")
print("="*60)

y_true_method5 = np.array(resultats_5["all_y_true_ind"][method_to_test])
y_pred_method5 = np.array(resultats_5["all_y_pred_ind"][method_to_test])

# Affichage de la matrice de confusion (voir utils.py)
afficher_matrice_confusion(y_true_method5, y_pred_method5, classes=list(range(1, 6)))

print("-"*40)

# Affichage du rapport de classification complet (precision, rappel, F1-score), (voir utils.py)
print("\nRapport de classification :")
afficher_classification_report(y_true_method5, y_pred_method5, classes=list(range(1, 6)))

# ============================================================
# TABLEAU COMPARATIF FINAL
# ============================================================
print("\n" + "="*60)
print("TABLEAU COMPARATIF FINAL")
print("="*60)

print("\n{:<25} {:>15} {:>15}".format("Méthode / Mode", "9 classes", "5 classes"))
print("-"*60)

# Méthodes individuelles
for m in methods:
    acc9 = np.mean(resultats_9["resultats_individuels"][m]) * 100
    acc5 = np.mean(resultats_5["resultats_individuels"][m]) * 100
    print(f"{m:<25} {acc9:>14.2f}% {acc5:>14.2f}%")

print("-"*60)

# Modes du vote majoritaire
for mode in ["majoritaire_simple", "majoritaire_pondere", "majoritaire_confiance"]:
    acc9 = np.mean(resultats_9["resultats_par_mode"][mode]) * 100
    acc5 = np.mean(resultats_5["resultats_par_mode"][mode]) * 100
    print(f"{mode:<25} {acc9:>14.2f}% {acc5:>14.2f}%")

print("\n" + "="*60)
print("FIN DES TESTS")
print("="*60)

# ============================
# ANALYSE DES CLASSES PROBLÉMATIQUES – 9 CLASSES
# ============================
print("\n\n" + "="*60)
print("ANALYSE DES CLASSES PROBLÉMATIQUES")
print("="*60)

# Calcul des métriques par classe pour le vote majoritaire
metriques_9 = {}
for c in range(1, 10):
    indices_c = np.where(y_true_all == c)[0]
    y_true_c = y_true_all[indices_c]
    y_pred_c = y_pred_all[indices_c]
    precision = np.sum(y_pred_c == y_true_c) / len(y_true_c) if len(y_true_c) > 0 else 0
    rappel = np.sum(y_pred_c == y_true_c) / len(y_true_c) if len(y_true_c) > 0 else 0
    metriques_9[c] = {'precision': precision, 'rappel': rappel}

print("\nClasses difficiles (rappel < 90%) avec vote majoritaire (9 classes):")
for c in range(1, 10):
    if metriques_9[c]['rappel'] < 0.9:
        m = metriques_9[c]
        print(f"  Classe {c}: Rappel={m['rappel']*100:.1f}%, Précision={m['precision']*100:.1f}%")

# Comparaison méthode choisie vs vote
metriques_method = {}
for c in range(1, 10):
    indices_c = np.where(y_true_method9 == c)[0]
    y_true_c = y_true_method9[indices_c]
    y_pred_c = y_pred_method9[indices_c]
    precision = np.sum(y_pred_c == y_true_c) / len(y_true_c) if len(y_true_c) > 0 else 0
    rappel = np.sum(y_pred_c == y_true_c) / len(y_true_c) if len(y_true_c) > 0 else 0
    metriques_method[c] = {'precision': precision, 'rappel': rappel}

print(f"\nComparaison rappel {method_to_test} vs Vote (9 classes):")
for c in range(1, 10):
    rappel_method = metriques_method[c]['rappel']
    rappel_vote = metriques_9[c]['rappel']
    diff = (rappel_vote - rappel_method) * 100
    symbole = "+" if diff >= 0 else ""
    print(f"  Classe {c}: {method_to_test}={rappel_method*100:.1f}% → Vote={rappel_vote*100:.1f}% ({symbole}{diff:.1f}%)")


# ============================
# ANALYSE DE L'IMPORTANCE DES MÉTHODES
# ============================
print("\n\n" + "="*60)
print("ANALYSE DE L'IMPORTANCE DES MÉTHODES")
print("="*60)

# On utilise le même split fixe que précédemment 
data_split = separer_train_test_par_echantillon(data, echantillons_test, all_methods=True) # Utilisation de la fonction utilitaire (voir utils.py)

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