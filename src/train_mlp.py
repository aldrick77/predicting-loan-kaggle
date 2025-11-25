import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# Config rapide
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
SUB_PATH = "data/raw/sample_submission.csv"

def train_mlp():
    print("--- Démarrage du Réseau de Neurones (MLP) ---")
    
    # 1. Chargement
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    sub = pd.read_csv(SUB_PATH)
    
    target_col = 'loan_paid_back'
    id_col = 'id'
    
    # Feature Engineering 'Light' (Les NN aiment la simplicité)
    # On ajoute juste les ratios principaux
    for df in [df_train, df_test]:
        df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['monthly_income'] = df['annual_income'] / 12
    
    X = df_train.drop(columns=[target_col, id_col])
    y = df_train[target_col]
    X_test = df_test.drop(columns=[id_col])
    
    # 2. Préparation des données (Crucial pour MLP)
    # On sépare numérique et catégoriel
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Pipeline Numérique : On remplit les trous (Median) + On met à l'échelle (StandardScaler)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline Catégoriel : On remplit (Missing) + OneHotEncoding (0/1)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    # 3. Le Modèle (2 couches cachées de 128 et 64 neurones)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,      # Régularisation L2
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=20,       # On fait peu d'époques pour aller vite (Early Stopping gérera)
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=True
    )
    
    # Pipeline Global
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', mlp)])
    
    # 4. Entraînement sur tout le dataset 
    print("Entraînement en cours (ça peut prendre 2-3 min)...")
    model_pipeline.fit(X, y)
    
    # Score rapide sur le subset de validation interne du MLP
    print(f"Score final (Loss): {model_pipeline.named_steps['classifier'].loss_}")
    
    # 5. Prédiction
    preds = model_pipeline.predict_proba(X_test)[:, 1]
    
    # Sauvegarde
    out_path = "outputs/submission_mlp.csv"
    submission = pd.DataFrame({'id': sub['id'], 'loan_paid_back': preds})
    submission.to_csv(out_path, index=False)
    print(f"✅ Submission MLP générée : {out_path}")

if __name__ == "__main__":
    train_mlp()