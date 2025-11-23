import numpy as np
from utils import (
    charger_donnees, 
    predire_classe, 
    separer_train_test_par_echantillon,
    separer_train_test_aleatoire,
    validation_croisee_knn,
    matrice_confusion,
    afficher_matrice_confusion,
    analyser_precision_par_classe
)

# Fixer la seed pour la reproductibilité
np.random.seed(42)

# Dossier principal contenant les sous-dossiers qui sont les methodes
base_folder = "C:/Users/lakaf/OneDrive/Bureau/25-26/RF/Projet_RF/data"

# Charger toutes les données
data = charger_donnees(base_folder)

# ============================
# DÉMONSTRATION DU PROBLÈME
# ============================

print("="*50)
print("DÉMONSTRATION DU PROBLÈME DE FUITE DE DONNÉES")
print("="*50)

# Préparer les données avec GFD
X_old = []
y_old = []
for key, value in data.items():
    if "GFD" in value:
        X_old.append(value["GFD"])
        y_old.append(value["class"])

X_old = np.array(X_old)
y_old = np.array(y_old)

# Test avec l'ancienne méthode (problématique)
print("\n1. ANCIENNE MÉTHODE (avec fuite de données) :")
X_train_old, y_train_old, X_test_old, y_test_old = separer_train_test_aleatoire(X_old, y_old, ratio_test=0.3)

y_pred_old = []
for x in X_test_old:
    y_pred_old.append(predire_classe(X_train_old, y_train_old, x, k=5))

accuracy_old = np.sum(np.array(y_pred_old) == y_test_old) / len(y_test_old)
print(f"Précision : {accuracy_old*100:.2f}%")
print("Cette précision est artificiellement élevée !")

"""
TEST DU KNN CORRIGÉ
"""

print("\n\n" + "="*50)
print("TEST KNN AVEC SÉPARATION CORRIGÉE")
print("="*50)

# Test avec différentes valeurs de k
for method in ["GFD", "E34"]:
    print(f"\n\nMÉTHODE : {method}")
    print("-"*30)
    
    for k in [3, 5, 7]:
        print(f"\nTest avec k={k} voisins")
        mean_acc, std_acc, accuracies = validation_croisee_knn(
            data, method, k=k, n_folds=3
        )
        
        print(f"\nRésumé pour k={k}:")
        print(f"Précisions par fold : {[f'{acc*100:.2f}%' for acc in accuracies]}")
        print(f"Précision moyenne : {mean_acc*100:.2f}% (±{std_acc*100:.2f}%)")

"""
ANALYSE DES ERREURS
"""

print("\n\n" + "="*50)
print("ANALYSE DÉTAILLÉE")
print("="*50)

# Test sur un fold pour analyser les erreurs
X_train, y_train, X_test, y_test = separer_train_test_par_echantillon(
    data, "GFD", [2, 5, 8]
)

y_pred = []
for x in X_test:
    y_pred.append(predire_classe(X_train, y_train, x, k=5))

y_pred = np.array(y_pred)

print("\nMatrice de confusion (lignes=vraies classes, colonnes=prédictions):")
confusion = matrice_confusion(y_test, y_pred)
afficher_matrice_confusion(confusion)

# Analyser les erreurs par classe
print("\nPrécision par classe:")
analyser_precision_par_classe(y_test, y_pred)
