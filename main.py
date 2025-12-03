import numpy as np
from data_loader import charger_donnees
from knn import validation_croisee_knn, predire_knn
from utils import separer_train_test_par_echantillon, afficher_matrice_confusion, afficher_classification_report
from kmeans import kmeans, evaluer_clustering, mapper_clusters_vers_classes, validation_croisee_kmeans

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

for method in methods:
    print(f"\nMéthode : {method}")
    
    # Séparer train/test
    X_train, y_train, X_test, y_test = separer_train_test_par_echantillon(data, method, echantillons_test)
    print(f"Échantillons de test : {echantillons_test}")
    
    # Prédiction
    y_pred = [predire_knn(X_train, y_train, x, k=5) for x in X_test]

    # Affichage de l'accuracy
    accuracy = np.mean(np.array(y_pred) == np.array(y_test))
    print(f"Accuracy : {accuracy*100:.2f}%\n")
    
    # Matrice de confusion
    print("Matrice de Confusion :")
    afficher_matrice_confusion(y_test, y_pred, classes, titre=f"KNN – Matrice de Confusion ({method})")
    
    # Rapport classification
    print("\nRapport de classification :\n")
    afficher_classification_report(y_test, y_pred, classes)


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
    for k in [1, 3, 5, 7]:
        mean_acc, std_acc, accs = validation_croisee_knn(X, y, k=k, n_folds=5)
        print(f"  k={k} → Moyenne : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  Accuracies par fold : {[f'{a*100:.2f}%' for a in accs]}")

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
    )

    print(f"  Moyenne : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Accuracies : {[f'{a*100:.2f}%' for a in accs]}")
