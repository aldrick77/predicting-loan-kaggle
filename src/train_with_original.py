import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from feature_engineering import create_features
import os

# Chemins
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
SAMPLE_PATH = "data/raw/sample_submission.csv"
ORIGINAL_PATH = "data/raw/original.csv"

def train_with_original():
    print("--- ☢️ NUCLEAR OPTION: Training with Original Data (CORRIGÉ) ☢️ ---")
    
    if not os.path.exists(ORIGINAL_PATH):
        print(f"❌ ERREUR : Tu dois télécharger le dataset original et le mettre dans {ORIGINAL_PATH}")
        return

    # 1. Chargement
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df_orig = pd.read_csv(ORIGINAL_PATH)
    
    print(f"Competition Train Shape: {df_train.shape}")
    print(f"Original Data Shape:     {df_orig.shape}")

    # --- FIX CRITIQUE : Gestion de la Target et de l'ID ---
    
    # A. Renommer la target de l'original si elle a un nom différent
    # Souvent dans ce dataset c'est 'loan_status', on vérifie
    if 'loan_paid_back' not in df_orig.columns:
        # On cherche une colonne qui ressemble à la target
        potential_targets = ['loan_status', 'Loan_Status', 'target']
        for col in potential_targets:
            if col in df_orig.columns:
                print(f"⚠️ Renommage de la colonne '{col}' en 'loan_paid_back'")
                df_orig = df_orig.rename(columns={col: 'loan_paid_back'})
                break
    
    # B. Création d'un faux ID pour l'original (pour matcher le format Kaggle)
    # On commence les IDs là où le train s'arrête pour éviter les doublons
    start_id = df_train['id'].max() + 1
    df_orig['id'] = range(start_id, start_id + len(df_orig))
    
    # 2. Harmonisation (Standardisation des colonnes)
    # On ne garde que les colonnes qui existent dans le train (sauf si l'original en manque)
    missing_cols = [c for c in df_train.columns if c not in df_orig.columns]
    if missing_cols:
        print(f"⚠️ Attention, colonnes manquantes dans l'original : {missing_cols}")
        # On les remplit avec des NaN ou on les drop si c'est critique
        for c in missing_cols:
            df_orig[c] = np.nan

    # Maintenant on force l'alignement strict
    df_orig = df_orig[df_train.columns]
    

    df_train['is_generated'] = 1
    df_test['is_generated'] = 1
    df_orig['is_generated'] = 0
    
    # 4. Fusion
    df_combined = pd.concat([df_train, df_orig], axis=0).reset_index(drop=True)
    print(f"--> Combined Training Set: {df_combined.shape}")
    
    # 5. Feature Engineering (V1 - La valeur sûre)
    df_combined = create_features(df_combined)
    df_test = create_features(df_test)
    
    # Gestion des manquants pour CatBoost
    cat_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
    # Sécurité: s'assurer que ces colonnes existent avant de fillna
    actual_cat_features = [c for c in cat_features if c in df_combined.columns]
    
    df_combined[actual_cat_features] = df_combined[actual_cat_features].fillna("Missing")
    df_test[actual_cat_features] = df_test[actual_cat_features].fillna("Missing")
    
    X = df_combined.drop(columns=['loan_paid_back', 'id'])
    y = df_combined['loan_paid_back']
    X_test = df_test.drop(columns=['id'])
    
    # 6. Entraînement CatBoost
    print("Démarrage de l'entraînement sur le dataset fusionné...")
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        eval_metric='AUC',
        random_seed=42,
        verbose=500,
        allow_writing_files=False,
        task_type="GPU"
    )
    
    model.fit(X, y, cat_features=actual_cat_features)
    
    # 7. Prédiction et Sauvegarde
    preds = model.predict_proba(X_test)[:, 1]
    
    out = "outputs/submission_with_original.csv"
    sub = pd.read_csv(SAMPLE_PATH)
    sub['loan_paid_back'] = preds
    sub.to_csv(out, index=False)
    print(f"✅ Submission Original Data générée : {out}")

if __name__ == "__main__":
    train_with_original()