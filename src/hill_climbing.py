import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

# Liste de TOUS nos modèles disponibles
MODELS = {
    'cat': {'oof': "outputs/oof_catboost.csv", 'sub': "outputs/submission_catboost_2025-11-25_03-29-09.csv"},
    'xgb': {'oof': "outputs/oof_xgboost.csv", 'sub': "outputs/submission_xgboost_2025-11-25_03-39-13.csv"},
    'lgb': {'oof': "outputs/oof_lightgbm.csv", 'sub': "outputs/submission_lightgbm_2025-11-25_03-41-44.csv"},
    'mlp': {'oof': "outputs/oof_mlp.csv", 'sub': "outputs/submission_mlp.csv"}
}

TRAIN_PATH = "data/raw/train.csv"
SAMPLE_SUB_PATH = "data/raw/sample_submission.csv"

def hill_climbing():
    print("--- 🧗 Démarrage du HILL CLIMBING (Expert Mode) ---")
    
    # 1. Chargement de la Vérité Terrain (Target)
    print("Chargement des données d'entraînement...")
    df_train = pd.read_csv(TRAIN_PATH)
    
    # CRUCIAL : On trie par ID pour garantir l'alignement
    df_train = df_train.sort_values('id').reset_index(drop=True)
    y_true = df_train['loan_paid_back'].values
    
    # On crée un set des IDs valides (ceux de la compétition d'origine)
    valid_ids = set(df_train['id'].values)
    print(f"Dataset de référence : {len(y_true)} lignes.")

    # 2. Chargement et Nettoyage des OOFs
    oofs = {}
    print("\nChargement des OOFs...")
    
    for name, paths in MODELS.items():
        if os.path.exists(paths['oof']):
            df = pd.read_csv(paths['oof'])
            
            # --- FIX TAILLE & ALIGNEMENT ---
            # 1. On ne garde que les IDs qui sont dans le train set officiel
            # (Ça vire les lignes du pseudo-labeling ou de l'original data en trop)
            df = df[df['id'].isin(valid_ids)]
            
            # 2. On trie par ID pour que la ligne 0 corresponde à la ligne 0 de y_true
            df = df.sort_values('id').reset_index(drop=True)
            
            # 3. Vérification de sécurité
            if len(df) != len(y_true):
                print(f"⚠️  Modèle {name} ignoré : Taille incorrecte ({len(df)} vs {len(y_true)})")
                continue
            
            # Si tout est bon, on garde les prédictions
            oofs[name] = df['pred'].values
            score = roc_auc_score(y_true, oofs[name])
            print(f"✅ Modèle {name}: CV AUC = {score:.5f}")
            
        else:
            print(f"⚠️  Fichier manquant pour {name}: {paths['oof']}")

    if not oofs:
        print("❌ ERREUR: Aucun modèle valide chargé. Vérifie tes fichiers.")
        return

    # 3. Algorithme Hill Climbing
    print("\nRecherche de la meilleure combinaison...")
    
    # On commence avec le meilleur modèle unique
    best_model_name = max(oofs, key=lambda k: roc_auc_score(y_true, oofs[k]))
    current_best_pred = oofs[best_model_name].copy()
    current_best_score = roc_auc_score(y_true, current_best_pred)
    
    # Initialisation des poids
    weights = {name: 0.0 for name in MODELS}
    weights[best_model_name] = 1.0
    
    # Paramètres de l'algo
    CYCLES = 50       # Nombre de fois qu'on essaie d'ajouter un modèle
    STEP = 0.1        # Poids ajouté à chaque étape
    
    print(f"Départ avec le champion : {best_model_name} (Score: {current_best_score:.5f})")
    
    for i in range(CYCLES):
        best_cycle_score = -1
        best_cycle_model = None
        
        # On teste l'ajout de chaque modèle disponible
        for name in oofs:
            # On ajoute une fraction (STEP) du modèle candidat au mélange actuel
            candidate_pred = current_best_pred + (STEP * oofs[name])
            score = roc_auc_score(y_true, candidate_pred)
            
            if score > best_cycle_score:
                best_cycle_score = score
                best_cycle_model = name
        
        # Si on a amélioré le score, on valide l'ajout
        if best_cycle_score > current_best_score:
            # print(f"Cycle {i+1}: Ajout de {best_cycle_model} -> New Score: {best_cycle_score:.6f}")
            current_best_pred = current_best_pred + (STEP * oofs[best_cycle_model])
            current_best_score = best_cycle_score
            weights[best_cycle_model] += STEP
    
    # 4. Normalisation des poids (pour que la somme fasse 1)
    total_weight = sum(weights.values())
    final_weights = {k: v / total_weight for k, v in weights.items()}
    
    # 5. Affichage des Résultats Locaux
    # Calcul du meilleur score individuel pour comparer
    best_single_score = max([roc_auc_score(y_true, p) for p in oofs.values()])
    
    print("\n" + "="*40)
    print(f"📊 RÉSULTATS LOCAUX (Cross-Validation)")
    print("="*40)
    print(f"Meilleur modèle seul       : {best_single_score:.5f}")
    print(f"Ensemble Hill Climbing     : {current_best_score:.5f}")
    print(f"Gain obtenu (Local)        : +{(current_best_score - best_single_score):.5f}")
    print("-" * 40)
    print("🏆 Poids Optimaux :")
    for k, v in final_weights.items():
        if v > 0.001: # On affiche que ceux qui servent
            print(f"  - {k.upper().ljust(8)} : {v:.4f}")
    print("="*40)

    # 6. Génération de la Soumission Finale
    print("\nGénération de la soumission finale...")
    final_sub_pred = 0
    
    # On charge un fichier modèle juste pour avoir les IDs du test
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    
    for name, w in final_weights.items():
        if w > 0:
            if os.path.exists(MODELS[name]['sub']):
                df_sub = pd.read_csv(MODELS[name]['sub'])
                final_sub_pred += w * df_sub['loan_paid_back']
            else:
                print(f"⚠️ Fichier submission manquant pour {name}, impossible de l'inclure !")
    
    out_path = "outputs/submission_hill_climbing.csv"
    submission = pd.DataFrame({'id': sample['id'], 'loan_paid_back': final_sub_pred})
    submission.to_csv(out_path, index=False)
    print(f"✅ Fichier prêt : {out_path}")

if __name__ == "__main__":
    hill_climbing()