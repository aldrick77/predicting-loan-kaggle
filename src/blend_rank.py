import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# On reprend les 3 fichiers de base (les champions individuels)
# Vérifie bien que ces fichiers existent toujours !
FILES = {
    'cat': "outputs/submission_catboost.csv", # 0.92279
    'lgb': "outputs/submission_lightgbm.csv", # 0.92182
    'xgb': "outputs/submission.csv"           # 0.92198
}

# Poids (On garde la logique qui a marché pour le Mega Blend)
WEIGHTS = {
    'cat': 0.50,
    'lgb': 0.30,
    'xgb': 0.20
}

def rank_blend():
    print("--- Démarrage du RANK AVERAGING ---")
    
    first = True
    df_blend = pd.DataFrame()
    
    for name, path in FILES.items():
        print(f"Traitement de {name}...")
        df = pd.read_csv(path)
        
        if first:
            df_blend['id'] = df['id']
            # Initialisation avec des zéros
            df_blend['rank_score'] = 0
            first = False
            
        # MAGIE DU RANK : On transforme les probas en classement (1, 2, 3... N)
        # pct=True met les rangs entre 0 et 1 pour normaliser
        df['rank'] = df['loan_paid_back'].rank(pct=True)
        
        # On ajoute au score global avec le poids
        df_blend['rank_score'] += df['rank'] * WEIGHTS[name]
    
    # On a maintenant un score de rang composite.
    # Pour faire propre, on le remet à l'échelle 0-1 (optionnel pour l'AUC mais mieux)
    scaler = MinMaxScaler()
    df_blend['loan_paid_back'] = scaler.fit_transform(df_blend[['rank_score']])
    
    # Nettoyage
    df_final = df_blend[['id', 'loan_paid_back']]
    
    out = "outputs/submission_rank_blend.csv"
    df_final.to_csv(out, index=False)
    print(f"✅ Fichier Rank Blend généré : {out}")
    print("Ce fichier est mathématiquement plus robuste que la moyenne simple.")

if __name__ == "__main__":
    rank_blend()