import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier  # <--- NOUVEAU
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import os
from feature_engineering import create_features

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Loan_Payback_Competition")
    
    with mlflow.start_run():
        mlflow.log_params(cfg.model.params)
        mlflow.log_param("model_name", cfg.model.name)
        
        # 1. Load Data
        print(f"Loading data...")
        df_train = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.train_path))
        df_test = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.test_path))
        sample_sub = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.submission_path))
        
        # 2. Feature Engineering
        df_train = create_features(df_train)
        df_test = create_features(df_test)
        
        # 3. Prep
        X = df_train.drop(columns=[cfg.training.target_col, cfg.training.id_col])
        y = df_train[cfg.training.target_col]
        X_test = df_test.drop(columns=[cfg.training.id_col])
        
        cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Catégories : {cat_features}")

        # 4. Encodage
        if cfg.model.name in ["xgboost", "lightgbm"]: # <--- LGBM rejoint XGBoost ici
            print("Encodage Label Encoding pour XGB/LGBM...")
            for col in cat_features:
                le = LabelEncoder()
                all_values = pd.concat([X[col], X_test[col]], axis=0).astype(str)
                le.fit(all_values)
                X[col] = le.transform(X[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                
        elif cfg.model.name == "catboost":
            print("Encodage Natif pour CatBoost...")
            X[cat_features] = X[cat_features].fillna("Missing")
            X_test[cat_features] = X_test[cat_features].fillna("Missing")

        # 5. Cross-Validation
        skf = StratifiedKFold(n_splits=cfg.training.n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        scores = []
        
        print(f"Démarrage du training avec {cfg.model.name}...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # --- BLOC MODELES ---
            if cfg.model.name == "xgboost":
                model = XGBClassifier(**cfg.model.params, early_stopping_rounds=50)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
            elif cfg.model.name == "lightgbm": # <--- NOUVEAU BLOC
                # LightGBM demande des callbacks pour le early_stopping maintenant, 
                # mais la méthode simple .fit marche encore souvent.
                # Pour être sûr, on utilise l'API sklearn standard
                model = LGBMClassifier(**cfg.model.params)
                # Astuce : LGBM peut être verbeux, on coupe les logs
                model.fit(
                    X_train, y_train, 
                    eval_set=[(X_val, y_val)],
                    eval_metric="auc",
                    callbacks=None # On gère manuellement si besoin, mais par défaut c'est ok
                )
                
            elif cfg.model.name == "catboost":
                model = CatBoostClassifier(**cfg.model.params)
                model.fit(
                    X_train, y_train,
                    cat_features=cat_features,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=50,
                    verbose=False
                )
            # --------------------

            val_p = model.predict_proba(X_val)[:, 1]
            test_p = model.predict_proba(X_test)[:, 1]

            oof_preds[val_idx] = val_p
            test_preds += test_p / cfg.training.n_splits
            
            score = roc_auc_score(y_val, val_p)
            scores.append(score)
            print(f"Fold {fold+1} AUC: {score:.5f}")
            mlflow.log_metric(f"auc_fold_{fold+1}", score)

        # 6. Fin
        overall_auc = roc_auc_score(y, oof_preds)
        print(f"\n--> Overall CV AUC ({cfg.model.name}): {overall_auc:.5f}")
        
        output_file = f"outputs/submission_{cfg.model.name}.csv"
        submission = pd.DataFrame({'id': sample_sub['id'], 'loan_paid_back': test_preds})
        submission.to_csv(output_file, index=False)
        print(f"Submission saved: {output_file}")

if __name__ == "__main__":
    train()