import pandas as pd

# Ton meilleur score ABSOLU (Le Mega Blend V2)
FILE_BEST = "outputs/submission_mega_blend.csv" # 0.92305
# Le nouveau challenger (MLP)
FILE_MLP = "outputs/submission_mlp.csv"

def galaxy_blend():
    df_best = pd.read_csv(FILE_BEST)
    df_mlp = pd.read_csv(FILE_MLP)
    
    # Stratégie : 90% sur les Arbres (Best), 10% sur le Neural Net
    # C'est suffisant pour corriger les erreurs systémiques sans casser le score
    w_best = 0.90
    w_mlp = 0.10
    
    print(f"Blending : {w_best} Best Tree Ensemble + {w_mlp} Neural Network")
    
    df_final = df_best.copy()
    df_final['loan_paid_back'] = (w_best * df_best['loan_paid_back']) + (w_mlp * df_mlp['loan_paid_back'])
    
    out = "outputs/submission_galaxy_blend.csv"
    df_final.to_csv(out, index=False)

if __name__ == "__main__":
    galaxy_blend()