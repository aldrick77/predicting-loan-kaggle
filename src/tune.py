import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from feature_engineering import create_features
import os
import yaml

# Configuration rapide
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
N_TRIALS = 20  # Nombre d'essais (Mets 50 ou 100 si tu as le temps, 20 pour tester vite)

def load_data():
    df_train = pd.read_csv(TRAIN_PATH)
    # On applique le même feature engineering que pour le training
    df_train = create_features(df_train)
    
    # Nettoyage basique pour CatBoost
    cat_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    df_train[cat_features] = df_train[cat_features].fillna("Missing")
    
    return df_train, cat_features

def objective(trial):
    # 1. Chargement des données (Optimisation: on pourrait le charger hors de la boucle)
    df_train, cat_features = load_data()
    X = df_train.drop(columns=['loan_paid_back', 'id'])
    y = df_train['loan_paid_back']

    # 2. L'espace de recherche (C'est ici qu'Optuna décide quoi tester)
    param = {
        'iterations': trial.suggest_int('iterations', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        # Paramètres fixes
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 0,
        'random_seed': 42,
        'task_type': 'GPU' # Change en 'GPU' si tu as une carte graphique Nvidia configurée
    }

    # 3. Cross-Validation Rapide (3 Folds pour aller plus vite pendant la recherche)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**param)
        
        # On utilise early_stopping pour ne pas perdre de temps
        model.fit(
            X_train, y_train,
            cat_features=cat_features,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        scores.append(score)

    return np.mean(scores)

if __name__ == "__main__":
    print("🚀 Démarrage de l'optimisation CatBoost avec Optuna...")
    
    # Création de l'étude (On cherche à MAXIMISER l'AUC)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n✅ Recherche terminée !")
    print(f"Meilleur score CV (approx): {study.best_value:.5f}")
    print("Meilleurs paramètres :")
    print(study.best_params)

    # Sauvegarde propre dans un fichier YAML
    best_params = study.best_params
    # On rajoute les params fixes importants pour la prod
    best_params['loss_function'] = 'Logloss'
    best_params['eval_metric'] = 'AUC'
    best_params['verbose'] = 100
    best_params['random_seed'] = 42
    
    with open("configs/model/catboost_tuned.yaml", "w") as f:
        yaml.dump({"name": "catboost", "params": best_params}, f)
    
    print("\n💾 Configuration sauvegardée dans 'configs/model/catboost_tuned.yaml'")
    print("Tu peux maintenant lancer : poetry run python src/training.py model=catboost_tuned")