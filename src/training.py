import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import os

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # 1. Setup MLflow
    # Utilisation d'une base de données SQLite locale au lieu de fichiers
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Loan_Payback_Baseline")
    
    with mlflow.start_run():
        # Log params
        mlflow.log_params(cfg.model.params)
        
        # 2. Chargement des données
        print(f"Chargement des données depuis {os.getcwd()}...")
        df_train = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.train_path))
        df_test = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.test_path))
        sample_sub = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.submission_path))
        
        # 3. Préparation simple (Baseline : on vire juste l'ID)
        X = df_train.drop(columns=[cfg.training.target_col, cfg.training.id_col])
        y = df_train[cfg.training.target_col]
        X_test = df_test.drop(columns=[cfg.training.id_col])
        
        # Gestion basique des strings (si jamais il y en a)
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            # Attention: pour la prod réelle, il faudrait fitter sur train et transform sur test proprement
            # Pour la baseline rapide, on suppose que test a les mêmes catégories ou on gère ça plus tard
            if col in X_test.columns:
                 X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'unknown') # Simplification
                 # On refit un LE simple pour l'instant pour éviter les erreurs, à améliorer v2
                 le_test = LabelEncoder()
                 X_test[col] = le_test.fit_transform(X_test[col].astype(str))

        # 4. Cross-Validation Stratifiée
        skf = StratifiedKFold(n_splits=cfg.training.n_splits, shuffle=True, random_state=42)
        
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        scores = []
        
        print(f"Démarrage du training ({cfg.model.name})...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # On passe early_stopping_rounds directement dans le constructeur
            # Note: On ajoute **cfg.model.params pour déballer les autres params du yaml
            model = XGBClassifier(
                **cfg.model.params,
                early_stopping_rounds=50
            )
            
            # Dans le fit, on retire l'argument fautif
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Prédictions
            val_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds
            test_preds += model.predict_proba(X_test)[:, 1] / cfg.training.n_splits
            
            score = roc_auc_score(y_val, val_preds)
            scores.append(score)
            print(f"Fold {fold+1} AUC: {score:.5f}")
            mlflow.log_metric(f"auc_fold_{fold+1}", score)

        # 5. Métriques Globales
        overall_auc = roc_auc_score(y, oof_preds)
        print(f"\n--> Overall CV AUC: {overall_auc:.5f}")
        print(f"Moyenne des scores: {np.mean(scores):.5f} (+/- {np.std(scores):.5f})")
        
        mlflow.log_metric("overall_auc", overall_auc)
        mlflow.log_metric("std_auc", np.std(scores))
        
        # 6. Sauvegarde Submission
        submission = pd.DataFrame({
            'id': sample_sub['id'],
            'loan_paid_back': test_preds
        })
        
        output_file = hydra.utils.to_absolute_path(cfg.data.output_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        submission.to_csv(output_file, index=False)
        print(f"Submission sauvegardée sous : {output_file}")

if __name__ == "__main__":
    train()