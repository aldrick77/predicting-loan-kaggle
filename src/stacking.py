import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

# Chemins des fichiers générés 
# OOF = Prédictions sur le Train 
OOF_FILES = {
    "xgb": "outputs/oof_xgboost.csv",
    "cat": "outputs/oof_catboost.csv",
    "lgb": "outputs/oof_lightgbm.csv"
}

# SUB = Prédictions sur le Test
SUB_FILES = {
    "xgb": "outputs/submission_xgboost.csv",
    "cat": "outputs/submission_catboost.csv",
    "lgb": "outputs/submission_lightgbm.csv"
}

TRAIN_PATH = "data/raw/train.csv"
SAMPLE_SUB_PATH = "data/raw/sample_submission.csv"

def stack():
    print("--- Démarrage du STACKING (Le Juge Arrive) ---")
    
    # 1. Chargement 
    print("Chargement des cibles réelles...")
    df_train = pd.read_csv(TRAIN_PATH)
    y_true = df_train['loan_paid_back']
    
    # 2. Construction du Dataset
    
    meta_X_train = pd.DataFrame()
    meta_X_test = pd.DataFrame()
    
    print("Assemblage des prédictions...")
    for model_name, path in OOF_FILES.items():
        if not os.path.exists(path):
            print(f"ERREUR CRITIQUE: Manque le fichier {path}")
            return
        
        # OOF (Train)
        df_oof = pd.read_csv(path)
        # Vérif alignement
        if not df_oof['id'].equals(df_train['id']):
            raise ValueError(f"Désalignement des IDs pour {model_name} !")
        meta_X_train[model_name] = df_oof['pred']
        
        # Test (Submission)
        sub_path = SUB_FILES[model_name]
        df_sub = pd.read_csv(sub_path)
        meta_X_test[model_name] = df_sub['loan_paid_back']

    print(f"Le Meta-Modèle va s'entraîner sur {meta_X_train.shape} données.")


    # On veut juste trouver les meilleurs coefficients a, b, c
    meta_model = LogisticRegression()
    meta_model.fit(meta_X_train, y_true)
    
    # Coefficients trouvés
    coeffs = meta_model.coef_[0]
    intercept = meta_model.intercept_[0]
    print("\n--- Verdict du Juge (Coefficients) ---")
    print(f"Intercept (Biais) : {intercept:.4f}")
    for name, coef in zip(OOF_FILES.keys(), coeffs):
        print(f"Confiance en {name.upper()} : {coef:.4f}")
        
    # 4. Prédiction Finale
    final_preds = meta_model.predict_proba(meta_X_test)[:, 1]
    
    # Score Interne du Stacking (Sur le train)
    # Attention, c'est un peu biaisé car on réutilise les OOF, mais ça donne une idée
    stack_score = roc_auc_score(y_true, meta_model.predict_proba(meta_X_train)[:, 1])
    print(f"\nScore estimé du Stacking (Train) : {stack_score:.5f}")

    # 5. Sauvegarde
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    submission = pd.DataFrame({'id': sample['id'], 'loan_paid_back': final_preds})
    
    out_path = "outputs/submission_stacking.csv"
    submission.to_csv(out_path, index=False)
    print(f"✅ Fichier Stacking prêt : {out_path}")

if __name__ == "__main__":
    stack()