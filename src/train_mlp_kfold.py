import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
SUB_PATH = "data/raw/sample_submission.csv"

def train_mlp_oof():
    print("--- Training MLP with 5-Fold CV for OOF ---")
    
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SUB_PATH)
    
    # Feature Engineering Light
    for df in [df_train, df_test]:
        df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['monthly_income'] = df['annual_income'] / 12

    X = df_train.drop(columns=['loan_paid_back', 'id'])
    y = df_train['loan_paid_back']
    X_test = df_test.drop(columns=['id'])

    # Preprocessing Pipeline
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    # K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Pipeline complet (re-créé à chaque fold pour éviter les fuites)
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42, early_stopping=True))
        ])
        
        model.fit(X_tr, y_tr)
        
        val_p = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_p
        test_preds += model.predict_proba(X_test)[:, 1] / 5
        
        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_p):.4f}")

    # Sauvegarde OOF
    df_oof = pd.DataFrame({'id': df_train['id'], 'pred': oof_preds})
    df_oof.to_csv("outputs/oof_mlp.csv", index=False)
    
    # Sauvegarde Submission
    sample_sub['loan_paid_back'] = test_preds
    sample_sub.to_csv("outputs/submission_mlp.csv", index=False)
    print("✅ OOF MLP Generated.")

if __name__ == "__main__":
    train_mlp_oof()