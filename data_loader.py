import os
import numpy as np

def charger_donnees(base_folder, classes, samples, methods):
    """
    Charge toutes les données depuis les fichiers .MET organisés par méthode.
    Retourne un dictionnaire : data[sXXnYYY] = {"class": int, "E34": vecteur, ...}
    """
    data = {}

    for c in classes:
        for s in samples:
            key = f"s{c:02d}n{s:03d}"
            data[key] = {"class": c} # ex : data["s01n001"] = {"class": 1}

            for method in methods:
                method_folder = os.path.join(base_folder, method)
                filename = os.path.join(method_folder, f"{key}.{method}") # ex : ./data/E34/s01n001.E34

                if os.path.exists(filename):
                    try:
                        with open(filename, "r") as f:
                            values = f.read().strip().split() # splitlines
                            data[key][method] = np.array(values, dtype=float)
                    except Exception as e:
                        print(f"Erreur lecture {filename} : {e}")
                else:
                    print(f"Fichier manquant : {filename}")
    return data
