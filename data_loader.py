import os
import numpy as np

def charger_donnees(base_folder, classes, samples, methods):
    """
    organisation des fichiers par méthode dans des sous-dossiers correspondant aux noms de méthodes, et chaque fichier doit être nommé selon le format sXXnYYY.METHOD.

    Args:
        base_folder: dossier racine contenant les sous-dossiers par méthode
        classes: liste ou range des numéros de classes (1 à 9)
        samples: liste ou range des numéros d'échantillons par classe
        methods: liste des méthodes à charger ["E34", "GFD",..]

    Returns:
        data: dictionnaire où chaque clé est un identifiant d'image
              (ex: "s01n001") et chaque valeur est un dictionnaire :
              {"class": numéro de classe, "E34": vecteur numpy, ...}
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
                            values = f.read().strip().split() # obtenir les valeurs separees
                            data[key][method] = np.array(values, dtype=float) # conversion en numpy array de float
                    except Exception as e:
                        print(f"Erreur lecture {filename} : {e}")
                else:
                    print(f"Fichier manquant : {filename}")
    return data
