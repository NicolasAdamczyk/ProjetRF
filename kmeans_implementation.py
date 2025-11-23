'''
Les résultats suggèrent que K-means a des difficultes avec la structure non-sphérique des données. 
Les classes 5 (main) et 8 (animal) mentionnées comme problématiques dans le PDF sont effectivement mal classées. 
Le clustering non-supervisé est inadapté pour ces formes complexes avec occlusions et distorsions.
Conclusion: Résultats cohérents avec les difficultés attendues mais performance trop faible pour être utilisable. 
KNN supervisé (98% avec GFD) largement supérieur.
'''

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from utils import charger_donnees, distance_euclidienne

# Fixer la seed pour la reproductibilité
np.random.seed(42)

# Dossier principal contenant les sous-dossiers qui sont les methodes
base_folder = "C:/Users/lakaf/OneDrive/Bureau/25-26/RF/Projet_RF/data"

# Charger toutes les données
data = charger_donnees(base_folder)

# ============================
# Implémentation K-Means
# ============================

def initialiser_centroides(X, k, method='random'):
    """
    Initialise k centroïdes
    method: 'random' ou 'kmeans++'
    """
    n = len(X)
    
    if method == 'random':
        # Sélection aléatoire de k échantillons comme centroïdes initiaux
        indices = np.random.choice(n, k, replace=False)
        return X[indices].copy()
    
    elif method == 'kmeans++':
        # Initialisation K-means++
        centroids = []
        
        # Premier centroïde aléatoire
        first_idx = np.random.randint(n)
        centroids.append(X[first_idx])
        
        for _ in range(1, k):
            # Calculer les distances au centroïde le plus proche
            distances = np.array([min([distance_euclidienne(x, c) for c in centroids]) for x in X])
            
            # Probabilités proportionnelles au carré des distances
            probabilities = distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            # Sélectionner le prochain centroïde
            next_idx = np.random.choice(n, p=probabilities)
            centroids.append(X[next_idx])
        
        return np.array(centroids)

def assigner_clusters(X, centroides):
    """
    Assigne chaque point au centroïde le plus proche
    Retourne les labels des clusters et les distances
    """
    n = len(X)
    k = len(centroides)
    labels = np.zeros(n, dtype=int)
    distances = np.zeros(n)
    
    for i in range(n):
        # Calculer la distance à chaque centroïde
        dists = [distance_euclidienne(X[i], centroides[j]) for j in range(k)]
        # Assigner au plus proche
        labels[i] = np.argmin(dists)
        distances[i] = min(dists)
    
    return labels, distances

def calculer_centroides(X, labels, k):
    """
    Recalcule les centroïdes comme la moyenne des points de chaque cluster
    """
    centroides = np.zeros((k, X.shape[1]))
    
    for i in range(k):
        # Sélectionner les points du cluster i
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            centroides[i] = np.mean(cluster_points, axis=0)
        else:
            # Si un cluster est vide, réinitialiser avec un point aléatoire
            centroides[i] = X[np.random.randint(len(X))]
    
    return centroides

def kmeans(X, k, max_iter=100, tol=1e-4, init='kmeans++', n_init=10):
    """
    Algorithme K-means complet
    
    Paramètres:
    - X: données
    - k: nombre de clusters
    - max_iter: nombre maximum d'itérations
    - tol: tolérance pour la convergence
    - init: méthode d'initialisation ('random' ou 'kmeans++')
    - n_init: nombre de fois à répéter avec différentes initialisations
    
    Retourne:
    - best_labels: meilleurs labels trouvés
    - best_centroids: meilleurs centroïdes
    - best_inertia: meilleure inertie (somme des distances au carré)
    """
    best_inertia = float('inf')
    best_labels = None
    best_centroids = None
    
    for run in range(n_init):
        # Initialiser les centroïdes
        centroides = initialiser_centroides(X, k, method=init)
        
        for iteration in range(max_iter):
            # Assigner les points aux clusters
            labels, distances = assigner_clusters(X, centroides)
            
            # Calculer les nouveaux centroïdes
            nouveaux_centroides = calculer_centroides(X, labels, k)
            
            # Vérifier la convergence
            if np.allclose(centroides, nouveaux_centroides, atol=tol):
                break
            
            centroides = nouveaux_centroides
        
        # Calculer l'inertie (somme des distances au carré)
        inertia = np.sum(distances ** 2)
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroides
    
    return best_labels, best_centroids, best_inertia

def evaluer_clustering(y_true, y_pred):
    """
    Évalue la qualité du clustering
    Calcule la pureté et l'indice de Rand ajusté
    """
    n = len(y_true)
    k = len(np.unique(y_pred))
    
    # Calculer la pureté
    purity = 0
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        if np.sum(mask) > 0:
            # Classe majoritaire dans ce cluster
            classe_majoritaire = Counter(y_true[mask]).most_common(1)[0][1]
            purity += classe_majoritaire
    purity = purity / n
    
    # Matrice de confusion pour le clustering
    confusion = np.zeros((k, 9), dtype=int)
    for i in range(n):
        confusion[y_pred[i], y_true[i] - 1] += 1
    
    return purity, confusion

def mapper_clusters_vers_classes(y_pred, y_true):
    """
    Trouve la meilleure correspondance entre clusters et vraies classes
    """
    clusters_uniques = np.unique(y_pred)
    mapping = {}
    
    for cluster in clusters_uniques:
        mask = y_pred == cluster
        if np.sum(mask) > 0:
            # Trouver la classe majoritaire pour ce cluster
            classe_majoritaire = Counter(y_true[mask]).most_common(1)[0][0]
            mapping[cluster] = classe_majoritaire
    
    # Créer les labels mappés
    y_mapped = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        y_mapped[i] = mapping[y_pred[i]]
    
    return y_mapped, mapping

# ============================
# Test du K-Means
# ============================

print("\n" + "="*50)
print("TEST K-MEANS")
print("="*50)

# Test avec différentes méthodes et valeurs de k
for method in ["GFD", "E34"]:
    print(f"\n\nMÉTHODE : {method}")
    print("-"*30)
    
    # Préparer les données
    X = []
    y_true = []
    keys = []
    
    for key, value in data.items():
        if method in value:
            X.append(value[method])
            y_true.append(value["class"])
            keys.append(key)
    
    X = np.array(X)
    y_true = np.array(y_true)
    
    print(f"Forme des données: {X.shape}")
    
    # Tester différentes valeurs de k
    for k in [9, 7, 11]:  # 9 = nombre réel de classes
        print(f"\n--- K-Means avec k={k} clusters ---")
        
        # Appliquer K-means
        y_pred, centroids, inertia = kmeans(X, k, max_iter=100, init='kmeans++', n_init=10)
        
        print(f"Inertie finale: {inertia:.2f}")
        
        # Évaluer
        purity, confusion = evaluer_clustering(y_true, y_pred)
        print(f"Pureté: {purity*100:.2f}%")
        
        # Mapper les clusters aux classes
        y_mapped, mapping = mapper_clusters_vers_classes(y_pred, y_true)
        accuracy = np.sum(y_mapped == y_true) / len(y_true)
        print(f"Précision après mapping: {accuracy*100:.2f}%")
        
        # Afficher le mapping
        print("\nMapping clusters -> classes:")
        for cluster, classe in sorted(mapping.items()):
            count = np.sum(y_pred == cluster)
            print(f"  Cluster {cluster} -> Classe {classe} ({count} échantillons)")
        
        # Distribution des clusters
        print("\nDistribution des tailles de clusters:")
        for i in range(k):
            count = np.sum(y_pred == i)
            print(f"  Cluster {i}: {count} échantillons")

# ============================
# Analyse de la stabilité
# ============================

print("\n\n" + "="*50)
print("ANALYSE DE LA STABILITÉ")
print("="*50)

method = "GFD"
k = 9

# Préparer les données
X = []
y_true = []
for key, value in data.items():
    if method in value:
        X.append(value[method])
        y_true.append(value["class"])

X = np.array(X)
y_true = np.array(y_true)

# Exécuter K-means plusieurs fois pour tester la stabilité
purities = []
inertias = []

print(f"\nTest de stabilité avec {method}, k={k}")
print("10 exécutions indépendantes:")

for i in range(10):
    y_pred, centroids, inertia = kmeans(X, k, max_iter=100, init='kmeans++', n_init=1)
    purity, _ = evaluer_clustering(y_true, y_pred)
    purities.append(purity)
    inertias.append(inertia)
    print(f"  Run {i+1}: Pureté={purity*100:.2f}%, Inertie={inertia:.2f}")

print(f"\nStatistiques sur 10 runs:")
print(f"  Pureté: moyenne={np.mean(purities)*100:.2f}%, std={np.std(purities)*100:.2f}%")
print(f"  Inertie: moyenne={np.mean(inertias):.2f}, std={np.std(inertias):.2f}")

# ============================
# Détermination du nombre optimal de clusters (méthode du coude)
# ============================

print("\n\n" + "="*50)
print("MÉTHODE DU COUDE")
print("="*50)

method = "GFD"

# Préparer les données
X = []
for key, value in data.items():
    if method in value:
        X.append(value[method])
X = np.array(X)

# Tester différentes valeurs de k
k_values = range(2, 16)
inertias = []
purities = []

print(f"\nCalcul pour différentes valeurs de k ({method}):")
for k in k_values:
    y_pred, centroids, inertia = kmeans(X, k, max_iter=100, init='kmeans++', n_init=5)
    purity, _ = evaluer_clustering(y_true, y_pred)
    inertias.append(inertia)
    purities.append(purity)
    print(f"  k={k:2d}: Inertie={inertia:8.2f}, Pureté={purity*100:.2f}%")

# Calculer le ratio d'amélioration
print("\nRatio d'amélioration de l'inertie:")
for i in range(1, len(inertias)):
    improvement = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
    print(f"  k={k_values[i-1]} -> k={k_values[i]}: {improvement:.2f}%")

# Suggestion du nombre optimal (basé sur le coude)
improvements = []
for i in range(1, len(inertias)):
    improvements.append(inertias[i-1] - inertias[i])

if len(improvements) > 1:
    # Chercher où l'amélioration diminue significativement
    for i in range(1, len(improvements)):
        if improvements[i] < improvements[i-1] * 0.5:  # Si l'amélioration chute de plus de 50%
            k_optimal = k_values[i]
            break
    else:
        k_optimal = 9  # Valeur par défaut

    print(f"\nNombre de clusters suggéré: {k_optimal}")
    print("(Basé sur la méthode du coude - point où l'amélioration ralentit)")

# ============================
# Comparaison avec les vraies classes
# ============================

print("\n\n" + "="*50)
print("ANALYSE DÉTAILLÉE DES ERREURS")
print("="*50)

method = "GFD"
k = 9

# Une dernière exécution pour l'analyse
X = []
y_true = []
keys = []

for key, value in data.items():
    if method in value:
        X.append(value[method])
        y_true.append(value["class"])
        keys.append(key)

X = np.array(X)
y_true = np.array(y_true)

y_pred, centroids, inertia = kmeans(X, k, max_iter=100, init='kmeans++', n_init=10)
y_mapped, mapping = mapper_clusters_vers_classes(y_pred, y_true)

# Identifier les échantillons mal classés
print("\nÉchantillons mal classés après mapping:")
erreurs = []
for i in range(len(y_true)):
    if y_mapped[i] != y_true[i]:
        erreurs.append((keys[i], y_true[i], y_mapped[i], y_pred[i]))

print(f"Nombre total d'erreurs: {len(erreurs)}")
print("\nDétail (limité aux 10 premières erreurs):")
for key, vraie, predite, cluster in erreurs[:10]:
    print(f"  {key}: Vraie classe={vraie}, Prédite={predite} (cluster {cluster})")

# Analyser les classes problématiques
print("\nClasses avec le plus d'erreurs:")
erreurs_par_classe = Counter([vraie for _, vraie, _, _ in erreurs])
for classe, count in erreurs_par_classe.most_common():
    total_classe = np.sum(y_true == classe)
    pct = count / total_classe * 100
    print(f"  Classe {classe}: {count}/{total_classe} erreurs ({pct:.1f}%)")
