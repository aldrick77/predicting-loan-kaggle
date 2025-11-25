import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from feature_engineering import create_features
import os

# Chemins
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
SAMPLE_PATH = "data/raw/sample_submission.csv"
ORIGINAL_PATH = "data/raw/original.csv"

def residual_boosting():
    print("--- 🚀 Démarrage du Boosting sur Résidus (Manual Gradient Boosting) ---")
    
    if not os.path.exists(ORIGINAL_PATH):
        print(f"❌ ERREUR : Il faut le fichier {ORIGINAL_PATH}")
        return

    # 1. Chargement des Données
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df_orig = pd.read_csv(ORIGINAL_PATH)
    
    # 2. Harmonisation de l'Original (Même logique que tout à l'heure)
    if 'loan_paid_back' not in df_orig.columns:
        cols_map = {'loan_status': 'loan_paid_back', 'Loan_Status': 'loan_paid_back'}
        df_orig = df_orig.rename(columns=cols_map)
    
    # On aligne les colonnes
    cols_to_keep = [c for c in df_train.columns if c != 'id' and c in df_orig.columns]
    df_orig = df_orig[cols_to_keep]
    
    # 3. Feature Engineering (Le même pour tout le monde)
    print("Application du Feature Engineering...")
    df_train = create_features(df_train)
    df_test = create_features(df_test)
    df_orig = create_features(df_orig)
    
    # Prep CatBoost (Gestion des NaNs)
    cat_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
    cat_features = [c for c in cat_features if c in df_train.columns]
    
    for df in [df_train, df_test, df_orig]:
        df[cat_features] = df[cat_features].fillna("Missing")

    # --- ÉTAPE 1 : Le Modèle "Base" sur l'Original ---
    print("\n--- Stage 1: Training on Original Data (The Teacher) ---")
    X_orig = df_orig.drop(columns=['loan_paid_back'])
    y_orig = df_orig['loan_paid_back']
    
    model_s1 = CatBoostClassifier(
        iterations=2000, 
        learning_rate=0.03, 
        depth=6, 
        cat_features=cat_features,
        verbose=0,
        allow_writing_files=False,
        random_seed=42,
        task_type="GPU"
    )
    model_s1.fit(X_orig, y_orig)
    
    # Prédictions du Stage 1 sur les données de la Compétition
    pred_s1_train = model_s1.predict_proba(df_train.drop(columns=['loan_paid_back', 'id']))[:, 1]
    pred_s1_test = model_s1.predict_proba(df_test.drop(columns=['id']))[:, 1]
    
    print("Stage 1 terminé.")

    # --- ÉTAPE 2 : Calcul des Résidus ---
    # Residual = Vérité - Prédiction Stage 1
    # Si le mec a remboursé (1) et qu'on a prédit 0.8 -> Erreur = 0.2 (Il faut ajouter 0.2)
    # Si le mec a pas remboursé (0) et qu'on a prédit 0.3 -> Erreur = -0.3 (Il faut enlever 0.3)
    y_train_comp = df_train['loan_paid_back']
    residuals = y_train_comp - pred_s1_train
    
    print(f"Statistiques des Résidus : Min={residuals.min():.3f}, Max={residuals.max():.3f}, Mean={residuals.mean():.3f}")

    # --- ÉTAPE 3 : Le Modèle "Correcteur" sur les Résidus ---
    print("\n--- Stage 2: Boosting over Residuals (The Student) ---")
    # ATTENTION : On utilise un REGRESSOR car on prédit une erreur continue, pas une classe !
    # On doit encoder les variables catégorielles pour XGBoost Regressor
    # (On fait un Label Encoding rapide ici pour simplifier, ou OneHot)
    from sklearn.preprocessing import OrdinalEncoder
    
    X_train_comp = df_train.drop(columns=['loan_paid_back', 'id'])
    X_test_comp = df_test.drop(columns=['id'])
    
    # Encodage pour XGBRegressor
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_comp[cat_features] = encoder.fit_transform(X_train_comp[cat_features].astype(str))
    X_test_comp[cat_features] = encoder.transform(X_test_comp[cat_features].astype(str))
    
    model_s2 = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        objective='reg:squarederror' # Régression !
    )
    
    model_s2.fit(X_train_comp, residuals)
    
    # Prédiction de la correction
    pred_correction = model_s2.predict(X_test_comp)
    
    # --- ÉTAPE 4 : Combinaison Finale ---
    print("\n--- Assemblage Final ---")
    # Finale = Prediction Base + Correction
    final_pred = pred_s1_test + pred_correction
    
    # Clipping : On s'assure de rester entre 0 et 1 (les maths peuvent déborder un peu)
    final_pred = np.clip(final_pred, 0, 1)
    
    # Sauvegarde
    out = "outputs/submission_residual_boost.csv"
    sub = pd.read_csv(SAMPLE_PATH)
    sub['loan_paid_back'] = final_pred
    sub.to_csv(out, index=False)
    print(f"✅ Submission Residual Boosting générée : {out}")

if __name__ == "__main__":
    residual_boosting()