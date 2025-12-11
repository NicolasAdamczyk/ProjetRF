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
              "s01n001" par exemple et chaque valeur est un dictionnaire :
              {"class": numéro de classe, "E34": vecteur numpy, ...}
    """
    data = {} #init dict pour stocker les donnees chargees

    #boucle sur chaque classe
    for c in classes:
        #boucle sur chaque echantillon (sample)
        for s in samples:
            #creation de la cle d identification de l image (ex:s01n001)
            key = f"s{c:02d}n{s:03d}" #formatage c sur 2 chiffres et s sur 3 chiffres
            data[key] = {"class": c} # ex : data["s01n001"] = {"class": 1}

            #boucle sur chaque methode a charger
            for method in methods:
                #construction du chemin du dossier de la methode
                method_folder = os.path.join(base_folder, method)
                #construction du chemin complet du fichier en utilisant la ligne du dessus
                filename = os.path.join(method_folder, f"{key}.{method}") # ex : ./data/E34/s01n001.E34

                #verif si le fichier existe
                if os.path.exists(filename):
                    #gestion erreur lors de la lecture
                    try:
                        #ouverture fichier en mode lecture
                        with open(filename, "r") as f:
                            values = f.read().strip().split() # obtenir les valeurs separees
                            #conversion de la liste de str en np.array de type float
                            data[key][method] = np.array(values, dtype=float) 
                    except Exception as e:
                        #affichage de l erreur si probleme de lecture/conversion
                        print(f"Erreur lecture {filename} : {e}")
                else:
                    #affichage si fichier manquant
                    print(f"Fichier manquant : {filename}")
    #retourne le dict data rempli
    return data
