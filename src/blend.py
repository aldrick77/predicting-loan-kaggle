import pandas as pd
import os

# Chemins vers tes fichiers de soumission (vérifie qu'ils sont bien dans outputs/)
# SUB_XGB = Ton fichier à 0.92198
# SUB_CAT = Ton fichier à 0.92279
SUB_XGB = "outputs/submission.csv"          
SUB_CAT = "outputs/submission_catboost.csv" 

def blend():
    print("--- Démarrage du Blending ---")
    
    # 1. Chargement
    if not os.path.exists(SUB_XGB) or not os.path.exists(SUB_CAT):
        print("ERREUR : Un des fichiers de soumission manque !")
        return

    df_xgb = pd.read_csv(SUB_XGB)
    df_cat = pd.read_csv(SUB_CAT)
    
    # 2. Sécurité : Vérifier que les IDs sont bien alignés
    # Si les IDs ne sont pas dans le même ordre, on casse tout.
    pd.testing.assert_series_equal(df_xgb['id'], df_cat['id'], obj="IDs")
    print("Check intégrité : Les IDs correspondent parfaitement.")
    
    # 3. La Formule Magique (Moyenne Pondérée)
    # CatBoost est meilleur (0.9228 vs 0.9220), on lui donne plus de poids.
    # Poids : 65% CatBoost / 35% XGBoost
    w_cat = 0.65
    w_xgb = 0.35
    
    df_blend = df_xgb.copy()
    df_blend['loan_paid_back'] = (w_xgb * df_xgb['loan_paid_back']) + (w_cat * df_cat['loan_paid_back'])
    
    # 4. Sauvegarde
    output_path = "outputs/submission_blend_v1.csv"
    df_blend.to_csv(output_path, index=False)
    
    print(f"\n✅ Fichier généré : {output_path}")
    print(f"Moyenne XGB      : {df_xgb['loan_paid_back'].mean():.4f}")
    print(f"Moyenne CatBoost : {df_cat['loan_paid_back'].mean():.4f}")
    print(f"Moyenne Blend    : {df_blend['loan_paid_back'].mean():.4f}")
    print("--> Tu peux soumettre ce fichier sur Kaggle !")

if __name__ == "__main__":
    blend()