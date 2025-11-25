import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = df.copy()
    
    # --- 1. NETTOYAGE PRÉALABLE ---
    # On remplace les 0 par 1 pour éviter les divisions par zéro
    df_feat['annual_income'] = df_feat['annual_income'].replace(0, 1)
    
    # --- 2. V2: FEATURES CIBLÉES (Basées sur le Diagnostic SHAP) ---

    # Le modèle surestime la sécurité des gens avec peu de dettes.
    # On va flagger ceux qui ont peu de dettes MAIS sont Chômeurs ou Étudiants.
    if 'debt_to_income_ratio' in df_feat.columns and 'employment_status' in df_feat.columns:
        # On considère "Low Debt" comme le quart le plus bas (DTI < ~0.10)
        # Note: On utilise transform pour calculer le quantile sur le dataset global si besoin, 
        # mais ici une valeur fixe approximative ou calculée est plus stable.
        # On va dire DTI < 0.15 (15%) c'est faible.
        
        risky_jobs = ['Unemployed', 'Student'] # Les profils instables
        
        # Création du flag binaire (1 = Profil "Piège", 0 = Normal)
        df_feat['flag_low_dti_risky_job'] = np.where(
            (df_feat['debt_to_income_ratio'] < 0.15) & 
            (df_feat['employment_status'].isin(risky_jobs)),
            1, 0
        )

    # B. Interaction Score vs Statut (Cross Feature)
    # Un score de 700 n'a pas la même valeur pour un Retraité (sûr) que pour un Chômeur (suspect).
    if 'credit_score' in df_feat.columns and 'employment_status' in df_feat.columns:
        # score en 5 catégories (Quantiles) pour réduire le bruit
        # qcut gère mieux la distribution que cut
        df_feat['score_bin'] = pd.qcut(df_feat['credit_score'], q=5, labels=False, duplicates='drop')
        
        # On crée l'interaction (ex: "4_Retired", "0_Unemployed")
        # CatBoost adore ce genre de feature combinée
        df_feat['score_status_interact'] = df_feat['score_bin'].astype(str) + "_" + df_feat['employment_status']
        
        # On supprime la colonne temporaire
        df_feat = df_feat.drop(columns=['score_bin'])

    # C. Ratio Prêt / Revenu (Le classique qui reste fort)
    # C'est la pression financière brute sur le ménage
    df_feat['loan_to_income_ratio'] = df_feat['loan_amount'] / df_feat['annual_income']
    
    # --- 3. NETTOYAGE FINAL (Suppression du bruit SHAP) ---
    # SHAP a montré que ces features n'aidaient pas ou étaient redondantes.
    # On les supprime pour que le modèle se concentre sur l'essentiel.
    useless_cols = [
        'gender',                   # Zéro impact
        'estimated_monthly_debt',   # Redondant avec DTI
        'total_interest_cost',      # Redondant avec Loan Amount
        'score_rate_interaction',   # Échec de la V1
        'monthly_income'            # Redondant avec Annual Income
    ]
    df_feat = df_feat.drop(columns=[c for c in useless_cols if c in df_feat.columns], errors='ignore')

    return df_feat