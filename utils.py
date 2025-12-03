import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc

def distance_euclidienne(a, b):
    """Calcule la distance euclidienne entre deux vecteurs."""
    return np.sqrt(np.sum((a - b) ** 2))

def separer_train_test_par_echantillon(data, method, echantillons_test):
    """Sépare le jeu de données en train/test selon des numéros d’échantillons complets."""
    X_train, y_train, X_test, y_test = [], [], [], []

    for key, value in data.items():
        if method in value:
            sample_num = int(key[4:7])
            if sample_num in echantillons_test:
                X_test.append(value[method])
                y_test.append(value["class"])
            else:
                X_train.append(value[method])
                y_train.append(value["class"])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
'''
def afficher_matrice_confusion(y_true, y_pred, classes, titre="Matrice de Confusion"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(titre)
    plt.show()
'''
    
def afficher_matrice_confusion(y_true, y_pred, classes, titre="Matrice de Confusion"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(f"\n{titre} :")
    # Affichage avec entêtes pour plus de lisibilité
    header = "    " + " ".join([f"{c:>3}" for c in classes])
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:>3} " + " ".join([f"{val:>3}" for val in row]))

def afficher_classification_report(y_true, y_pred, classes):
    """
    Affiche précision, rappel et F1-score pour chaque classe.
    """
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))


'''
def courbe_precision_rappel_knn(y_test, y_pred_votes, classes, method_name, k):
    """
    Trace les courbes précision–rappel.
    y_pred_votes doit contenir pour chaque échantillon :
        - la classe prédite
        - le nombre de votes pour cette classe
    """

    plt.figure(figsize=(8,6))

    for c in classes:
        # création du vrai label binaire ("one vs rest")
        y_true_bin = np.array([1 if y == c else 0 for y in y_test])

        # score de confiance = votes / k
        y_scores = np.array([
            votes / k if pred == c else 0
            for pred, votes in y_pred_votes
        ])

        precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
        auc_score = auc(recall, precision)

        plt.plot(recall, precision, label=f"Classe {c} (AUC={auc_score:.2f})")

    plt.xlabel("Rappel")
    plt.ylabel("Précision")
    plt.title(f"Courbes PR – {method_name} – KNN (k={k})")
    plt.legend()
    plt.grid(True)
    plt.show()
'''
