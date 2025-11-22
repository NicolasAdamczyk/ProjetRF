import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def courbe_precision_rappel_knn(X_train, y_train, X_test, y_test, k=10, classe_cible=1):
    """
    Calcule et trace la courbe précision/rappel pour une classe cible du KNN.
    Pour KNN, on approxime la "probabilité" en comptant la proportion de voisins de chaque classe.
    """
    y_scores = []
    y_true_bin = (y_test == classe_cible).astype(int)

    # Calcul du "score" (proportion de voisins de la classe cible)
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        voisins_idx = np.argsort(distances)[:k]
        classes_voisins = y_train[voisins_idx]
        proportion = np.sum(classes_voisins == classe_cible) / k
        y_scores.append(proportion)

    y_scores = np.array(y_scores)

    # Calcul des points de la courbe
    precision, recall, seuils = precision_recall_curve(y_true_bin, y_scores)
    auc_pr = average_precision_score(y_true_bin, y_scores)

    # Affichage
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Rappel')
    plt.ylabel('Précision')
    plt.title(f'Courbe Précision/Rappel – Classe {classe_cible} (K={k}) – AUC={auc_pr:.3f}')
    plt.grid(True)
    plt.show() # Faut ouvrir sur un ide pour voir le plt sur un navigateur ca ouvre pas

    print(f"Aire sous la courbe (AUC) pour la classe {classe_cible} : {auc_pr:.3f}")
    return auc_pr
