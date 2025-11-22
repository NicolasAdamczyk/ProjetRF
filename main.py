import numpy as np
from data_loader import charger_donnees
from knn import validation_croisee_knn, predire_knn
from metrics import courbe_precision_rappel_knn
from utils import separer_train_test_par_echantillon, afficher_matrice_confusion, afficher_classification_report
from kmeans import kmeans, evaluer_clustering, mapper_clusters_vers_classes

# ============================
# CONFIGURATION
# ============================
np.random.seed(42)  # reproductibilité
base_folder = "./data"
classes = range(1, 10)  # s01 à s09
samples = range(1, 12)  # n001 à n011
methods = ["E34", "GFD", "SA", "F0", "F2"]

# ============================
# CHARGEMENT DES DONNÉES
# ============================
data = charger_donnees(base_folder, classes, samples, methods)
print("Lecture terminée.")
print("Nombre total d'images :", len(data))

# ============================
# TEST KNN SUR UN SIMPLE SPLIT FIXE
# ============================
print("\n" + "="*50)
print("TEST KNN SIMPLE SUR SPLIT FIXE")
print("="*50)

echantillons_test = [2, 5, 8]
knn_results = {}

for method in methods:
    print(f"\nMéthode : {method}")
    # Séparation train/test
    X_train, y_train, X_test, y_test = separer_train_test_par_echantillon(data, method, echantillons_test)
    
    # Prédiction KNN pour tous les vecteurs test
    y_pred = []
    y_votes = []

    for x in X_test:
        classe = predire_knn(X_train, y_train, x, k=5)
        y_pred.append(classe)


    knn_results[method] = {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred
    }

# ============================
# TEST KNN AVEC VALIDATION CROISÉE
# ============================
print("\n" + "="*50)
print("VALIDATION CROISÉE KNN")
print("="*50)

# Pour chaque méthode et chaque k
for method in methods:
    # Préparer X et y pour toute la méthode
    X = [data[key][method] for key in data if method in data[key]]
    y = [data[key]["class"] for key in data if method in data[key]]

    print(f"\nMéthode : {method}")
    for k in [3, 5, 7]:
        mean_acc, std_acc, accs = validation_croisee_knn(X, y, k=k, n_folds=5)
        print(f"  k={k} → Moyenne : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  Accuracies par fold : {[f'{a*100:.2f}%' for a in accs]}")

# ============================
# ANALYSE DE LA METHODE KNN
# ============================
print("\n" + "="*50)
print("ANALYSE DES ERREURS (GFD, k=5)")
print("="*50)

classes_bdshape = list(range(1,10))

for method in methods:
    print("\n" + "="*50)
    print(f"ANALYSE KNN – Méthode : {method}")
    print("="*50)

    y_test = knn_results[method]["y_test"]
    y_pred = knn_results[method]["y_pred"]

    # Matrice de confusion
    afficher_matrice_confusion(y_test, y_pred, classes_bdshape, titre=f"KNN – Matrice de Confusion ({method})")
    
    # Précision, rappel, F1-score
    afficher_classification_report(y_test, y_pred, classes_bdshape)

"""
# ============================
# COURBE PRECISION–RAPPEL KNN
# ============================
print("\n" + "="*50)
print("COURBES PRECISION–RAPPEL KNN")
print("="*50)

for method in methods:
    print(f"\nCourbe PR – Méthode : {method}")
    
    y_test = knn_results[method]["y_test"]
    y_pred_votes = knn_results[method]["y_votes"]
    
    courbe_precision_rappel_knn(
        y_test,
        y_pred_votes,
        classes_bdshape,
        method_name=method,
        k=5
    )
"""

# ============================
# TEST K-MEANS
# ============================
print("\n" + "="*50)
print("TEST K-MEANS")
print("="*50)

kmeans_results = {}

for method in methods:
    print(f"\nMéthode : {method}")
    
    # Préparer les données
    X = np.array([data[key][method] for key in data if method in data[key]])
    y_true = np.array([data[key]["class"] for key in data if method in data[key]])
    keys = [key for key in data if method in data[key]]

    # Paramètre k (nombre de clusters = nombre de classes pour commencer)
    k = 9

    # Appliquer K-Means
    y_pred, centroids, inertia = kmeans(X, k, max_iter=100, init='kmeans++', n_init=10)

    # Évaluer
    purity, confusion = evaluer_clustering(y_true, y_pred)
    y_mapped, mapping = mapper_clusters_vers_classes(y_pred, y_true)
    accuracy = np.sum(y_mapped == y_true) / len(y_true)

    print(f"Inertie: {inertia:.2f}, Pureté: {purity*100:.2f}%, Accuracy: {accuracy*100:.2f}%")

    # Matrice de confusion
    classes_bdshape = list(range(1, 10))
    afficher_matrice_confusion(y_true, y_mapped, classes_bdshape, titre=f"K-Means – Matrice de Confusion ({method})")
    
    # Précision, rappel, F1-score
    afficher_classification_report(y_true, y_mapped, classes_bdshape)

    # Sauvegarder résultats, pour un eventuel excel
    kmeans_results[method] = {
        "X": X,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_mapped": y_mapped,
        "centroids": centroids,
        "inertia": inertia,
        "purity": purity,
        "mapping": mapping,
        "keys": keys
    }
