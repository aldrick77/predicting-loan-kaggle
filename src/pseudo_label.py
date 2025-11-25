import pandas as pd
import numpy as np
import os

# Chemins
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
BEST_SUB = "outputs/submission_mega_blend.csv" # Ton meilleur fichier (0.92305)
OUTPUT_PATH = "data/processed/train_pseudo.csv"

# Seuils de confiance (Confidence Thresholds)
# On ne prend que les prédictions EXTRÊMES pour limiter le risque d'erreur
# Tu peux ajuster : 0.99/0.01 est très prudent. 0.95/0.05 est plus agressif.
UPPER_THRESHOLD = 0.99
LOWER_THRESHOLD = 0.01

def create_pseudo_label_dataset():
    print("--- Génération du Pseudo-Labeled Dataset ---")
    
    # 1. Chargement
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df_sub = pd.read_csv(BEST_SUB)
    
    print(f"Train original : {df_train.shape}")
    
    # 2. Identification des 'Easy Samples' dans le Test
    # On colle la prédiction à côté des features du test
    df_test['loan_paid_back'] = df_sub['loan_paid_back']
    
    # On filtre ceux qui sont sûrs
    confident_positive = df_test[df_test['loan_paid_back'] >= UPPER_THRESHOLD].copy()
    confident_negative = df_test[df_test['loan_paid_back'] <= LOWER_THRESHOLD].copy()
    
    # On force les labels à 1 ou 0 (Hard Labeling)
    confident_positive['loan_paid_back'] = 1
    confident_negative['loan_paid_back'] = 0
    
    print(f"Confidents Positifs (> {UPPER_THRESHOLD}) : {len(confident_positive)}")
    print(f"Confidents Négatifs (< {LOWER_THRESHOLD}) : {len(confident_negative)}")
    
    # 3. Fusion
    df_pseudo = pd.concat([df_train, confident_positive, confident_negative], axis=0)
    
    # On mélange pour que le training se passe bien
    df_pseudo = df_pseudo.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Nouveau Train shape : {df_pseudo.shape} (+ {len(df_pseudo) - len(df_train)} lignes)")
    
    # 4. Sauvegarde
    os.makedirs("data/processed", exist_ok=True)
    df_pseudo.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Fichier généré : {OUTPUT_PATH}")

if __name__ == "__main__":
    create_pseudo_label_dataset()