import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from feature_engineering import create_features
import os

# Config
TRAIN_PATH = "data/raw/train.csv"

def diagnose():
    print("--- 🩺 DIAGNOSTIC DU MODÈLE (Pourquoi on plafonne ?) ---")
    
    # 1. Chargement & Feature Engineering V1
    df = pd.read_csv(TRAIN_PATH)
    df = create_features(df) 
    
    # Nettoyage basique
    cat_features = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
    df[cat_features] = df[cat_features].fillna("Missing")
    
    X = df.drop(columns=['loan_paid_back', 'id'])
    y = df['loan_paid_back']
    
    # On garde 20% pour valider et analyser les erreurs
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 2. Entraînement Rapide (mais précis)
    print("Entraînement de la sonde...")
    model = CatBoostClassifier(
        iterations=1500, 
        learning_rate=0.05, 
        depth=6, 
        cat_features=cat_features,
        verbose=0,
        allow_writing_files=False,
        random_seed=42,
        processing = 'GPU'
    )
    model.fit(X_train, y_train)
    
    # 3. Calcul des Résidus (Erreurs)
    preds = model.predict_proba(X_val)[:, 1]
    df_val = X_val.copy()
    df_val['true'] = y_val
    df_val['pred'] = preds
    df_val['error'] = np.abs(df_val['true'] - df_val['pred'])
    
    # On regarde les 500 pires erreurs
    worst = df_val.sort_values('error', ascending=False).head(500)
    
    print(f"\n--- 1. ANALYSE DES PIRES ERREURS (Top 500) ---")
    print(f"Moyenne Erreur Globale : {df_val['error'].mean():.4f}")
    print(f"Moyenne Erreur (Pires) : {worst['error'].mean():.4f}")
    
    print("\nQui sont ces clients incompris ? (Moyennes)")
    # On compare la moyenne des pires erreurs vs la moyenne globale
    # Si une variable explose, c'est elle la coupable
    numeric_cols = ['annual_income', 'loan_amount', 'credit_score', 'loan_to_income_ratio', 'debt_to_income_ratio']
    
    res = pd.DataFrame()
    res['Global_Avg'] = df_val[numeric_cols].mean()
    res['Worst_Errors_Avg'] = worst[numeric_cols].mean()
    res['Diff_%'] = ((res['Worst_Errors_Avg'] - res['Global_Avg']) / res['Global_Avg']) * 100
    print(res.sort_values('Diff_%', ascending=False))

    # 4. SHAP VALUES (L'explication visuelle)     print("\n--- 2. IMPORTANCE DES FEATURES (SHAP) ---")
    print("Calcul en cours (ça peut prendre 1 min)...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    
    # On génère le graph
    plt.figure(figsize=(10, 12))
    shap.summary_plot(shap_values, X_val, show=False, max_display=20)
    plt.title("Quelles features pilotent la décision ?")
    
    output_img = "outputs/shap_summary.png"
    plt.savefig(output_img, bbox_inches='tight')
    print(f"✅ Graphique SHAP sauvegardé sous : {output_img}")
    print("Ouvre cette image pour voir quelles features sont inutiles !")

if __name__ == "__main__":
    diagnose()