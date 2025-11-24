import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère de nouvelles features financières pour enrichir le dataset.
    """
    df_feat = df.copy()
    
    # Éviter les divisions par zéro
    df_feat['annual_income'] = df_feat['annual_income'].replace(0, 1)
    
    # 1. Ratio Prêt / Revenu Annuel (Risque pur)
    df_feat['loan_to_income_ratio'] = df_feat['loan_amount'] / df_feat['annual_income']
    
    # 2. Coût total des intérêts (Estimation simplifiée)
    df_feat['total_interest_cost'] = df_feat['loan_amount'] * df_feat['interest_rate']
    
    # 3. Revenu mensuel estimé
    df_feat['monthly_income'] = df_feat['annual_income'] / 12
    
    # 4. Estimation de la mensualité de dette existante (basée sur DTI)
    # DTI = Total Monthly Debt / Gross Monthly Income
    # Donc Monthly Debt = DTI * Monthly Income
    df_feat['estimated_monthly_debt'] = df_feat['debt_to_income_ratio'] * df_feat['monthly_income']
    
    # 5. Interaction Score Crédit & Taux (Le taux est-il juste par rapport au score ?)
    # Un score haut avec un taux haut = suspect
    df_feat['score_rate_interaction'] = df_feat['credit_score'] * df_feat['interest_rate']

    return df_feat