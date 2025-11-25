import pandas as pd
import numpy as np
import glob
import os

def mass_average():
    print("--- 🐘 Démarrage du MASS AVERAGING ---")
    
    # 1. Récupération des fichiers
    # On cherche tous les CSV qui commencent par "submission_" dans le dossier outputs
    pattern = "outputs/submission_*.csv"
    files = glob.glob(pattern)
    
    # Si on ne trouve rien, on arrête tout de suite avec un message clair
    if not files:
        print(f"❌ AUCUN FICHIER TROUVÉ avec le motif : {pattern}")
        print("Vérifie que tu as bien des fichiers .csv dans le dossier 'outputs/'")
        return

    print(f"✅ Trouvé {len(files)} fichiers à fusionner :")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # 2. Chargement et lecture
    preds = []
    # On garde l'ID du premier fichier pour la soumission finale
    first_df = pd.read_csv(files[0])
    ids = first_df['id']
    
    for f in files:
        df = pd.read_csv(f)
        # Sécurité : vérifier que les IDs correspondent (taille)
        if len(df) != len(ids):
            print(f"⚠️ Attention : {f} a une taille différente ({len(df)}), ignoré.")
            continue
        preds.append(df['loan_paid_back'].values)
        
    # 3. Moyenne Simple (Axis 0 = moyenne colonne par colonne)
    print("\nCalcul de la moyenne...")
    avg_preds = np.mean(preds, axis=0)
    
    # 4. Sauvegarde
    submission = pd.DataFrame({
        'id': ids,
        'loan_paid_back': avg_preds
    })
    
    out = "outputs/submission_mass_avg.csv"
    submission.to_csv(out, index=False)
    print(f"🎉 Fichier Mass Avg généré : {out}")

# C'EST CETTE PARTIE QUI MANQUAIT PROBABLEMENT :
if __name__ == "__main__":
    mass_average()