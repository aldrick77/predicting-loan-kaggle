import pandas as pd

# Tes deux meilleurs atouts
FILE_SAFE = "outputs/submission_mega_blend.csv"      # Ton record (0.92305)
FILE_AGGRO = "outputs/submission_catboost.csv" # Le nouveau (Pseudo-Labeling)

def final_blend():
    print("--- Démarrage du FINAL MONSTER BLEND ---")
    
    df_safe = pd.read_csv(FILE_SAFE)
    df_aggro = pd.read_csv(FILE_AGGRO)
    
    # Vérification
    pd.testing.assert_series_equal(df_safe['id'], df_aggro['id'])
    
    # La Stratégie : 
    # On fait confiance majoritairement à notre record actuel (Safe)
    # Mais on laisse le Pseudo-Labeling (Aggro) corriger les cas limites
    w_safe = 0.70
    w_aggro = 0.30
    
    print(f"Mélange : {w_safe*100}% Safe Record + {w_aggro*100}% Pseudo-Labeling")
    
    df_final = df_safe.copy()
    df_final['loan_paid_back'] = (w_safe * df_safe['loan_paid_back']) + (w_aggro * df_aggro['loan_paid_back'])
    
    output_path = "outputs/submission_final_push.csv"
    df_final.to_csv(output_path, index=False)
    
    print(f"✅ Fichier FINAL généré : {output_path}")

if __name__ == "__main__":
    final_blend()