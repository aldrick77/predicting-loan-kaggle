import pandas as pd
import os

# Tes 3 champions
SUB_CAT = "outputs/submission_catboost.csv"  # 0.9223
SUB_LGB = "outputs/submission_lightgbm.csv"  # 0.9218
SUB_XGB = "outputs/submission.csv"           # 0.9216 (vérifie que c'est bien lui)

def mega_blend():
    print("--- Démarrage du MEGA BLEND (Cat + LGBM + XGB) ---")
    
    # 1. Chargement
    df_cat = pd.read_csv(SUB_CAT)
    df_lgb = pd.read_csv(SUB_LGB)
    df_xgb = pd.read_csv(SUB_XGB)
    
    # Sécurité ID
    pd.testing.assert_series_equal(df_cat['id'], df_lgb['id'])
    pd.testing.assert_series_equal(df_cat['id'], df_xgb['id'])
    
    # 2. Pondération (La recette de cuisine)
    # On donne plus de poids aux modèles les plus performants
    w_cat = 0.50
    w_lgb = 0.30
    w_xgb = 0.20
    
    print(f"Poids appliqués : CatBoost={w_cat}, LGBM={w_lgb}, XGB={w_xgb}")
    
    df_blend = df_cat.copy()
    df_blend['loan_paid_back'] = (
        (w_cat * df_cat['loan_paid_back']) + 
        (w_lgb * df_lgb['loan_paid_back']) + 
        (w_xgb * df_xgb['loan_paid_back'])
    )
    
    # 3. Sauvegarde
    output_path = "outputs/submission_mega_blend.csv"
    df_blend.to_csv(output_path, index=False)
    
    print(f"\n✅ Fichier généré : {output_path}")
    print(f"Moyenne finale du blend : {df_blend['loan_paid_back'].mean():.4f}")

if __name__ == "__main__":
    mega_blend()