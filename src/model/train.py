"""
Entraînement du modèle — Diabetes 130-US Hospitals
====================================================
Baseline (Logistic Regression) + modèle final (XGBoost), avec gestion
du déséquilibre de classes. Évalue avec AUC-ROC, AUC-PR et recall,
plus adaptés qu'une simple accuracy vu que la cible est déséquilibrée
(~9% de réadmissions <30j).

Usage:
    python src/model/train.py
"""

import sys
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

# Permet d'importer src.features quand le script est lancé directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.build_features import load_and_build_features

MODEL_DIR = "models"
RANDOM_STATE = 42


def find_best_threshold_f1(y_true, y_proba):
    """Choisit un seuil sur validation pour maximiser le F1-score
    (compromis équilibré précision/recall)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return threshold, float(f1_scores[best_idx])


def find_threshold_for_recall(y_true, y_proba, target_recall: float = 0.60):
    """Choisit le seuil le plus élevé (donc le plus précis) qui atteint
    au moins `target_recall`. Utile en contexte médical où rater un
    patient à risque (faux négatif) coûte plus cher qu'une fausse alerte.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # thresholds a une longueur de len(precision) - 1
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        # Aucun seuil n'atteint ce recall : on prend le seuil minimal
        return float(thresholds[0]), float(precision[0]), float(recall[0])
    # Parmi les seuils valides, on prend celui qui maximise la précision
    best_idx = valid_idx[np.argmax(precision[valid_idx])]
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def evaluate(model, X_test, y_test, name: str, threshold: float = 0.5):
    """Affiche les métriques adaptées à une cible déséquilibrée."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(f"AUC-ROC : {auc_roc:.4f}")
    print(f"AUC-PR  : {auc_pr:.4f}  (baseline aléatoire ≈ {y_test.mean():.4f})")
    print(f"Seuil retenu : {threshold:.3f}")
    print("\nClassification report :")
    print(classification_report(y_test, y_pred, target_names=["Non réadmis", "Réadmis <30j"]))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "threshold": threshold}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_and_build_features()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(
        f"\n[SPLIT] Train : {len(X_train)} lignes | Validation : {len(X_val)} lignes | Test : {len(X_test)} lignes"
    )

    # ------------------------------------------------------------------
    # Baseline — Logistic Regression avec scaling
    # ------------------------------------------------------------------
    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )
    baseline.fit(X_train, y_train)
    baseline_val_proba = baseline.predict_proba(X_val)[:, 1]
    baseline_threshold, _ = find_best_threshold_f1(y_val, baseline_val_proba)
    evaluate(baseline, X_test, y_test, "Baseline — Logistic Regression", baseline_threshold)

    # ------------------------------------------------------------------
    # Modèle final — XGBoost (avec early stopping)
    # ------------------------------------------------------------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n[XGBOOST] scale_pos_weight = {scale_pos_weight:.2f}")

    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=4,
        gamma=0.2,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,   # <-- stoppe si pas d'amélioration après 50 itérations
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,  # affiche 1 ligne toutes les 50 itérations au lieu de toutes les lignes
    )
    print(f"[XGBOOST] Meilleure itération : {xgb.best_iteration} "
          f"(sur {xgb.n_estimators} configurées) — AUC-PR val : {xgb.best_score:.4f}")

    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]

    # --- Seuil A : compromis équilibré (F1) ---
    xgb_threshold_f1, _ = find_best_threshold_f1(y_val, xgb_val_proba)
    xgb_metrics_f1 = evaluate(xgb, X_test, y_test, "XGBoost — seuil F1-optimal", xgb_threshold_f1)

    # --- Seuil B : orienté recall (viser au moins 60% des patients à risque détectés) ---
    xgb_threshold_recall, prec_at_recall, recall_at_recall = find_threshold_for_recall(
        y_val, xgb_val_proba, target_recall=0.60
    )
    print(f"\n[SEUIL RECALL] Sur validation : seuil={xgb_threshold_recall:.3f} "
          f"-> precision≈{prec_at_recall:.3f}, recall≈{recall_at_recall:.3f}")
    xgb_metrics_recall = evaluate(
        xgb, X_test, y_test, "XGBoost — seuil orienté recall (>=60%)", xgb_threshold_recall
    )

    # ------------------------------------------------------------------
    # Sauvegarde du modèle final + métadonnées (pour l'app Streamlit)
    # ------------------------------------------------------------------
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_columns.joblib"))
    joblib.dump(
        {
            "threshold_f1": xgb_metrics_f1["threshold"],
            "threshold_recall60": xgb_metrics_recall["threshold"],
            "auc_roc": xgb_metrics_f1["auc_roc"],
            "auc_pr": xgb_metrics_f1["auc_pr"],
            "best_iteration": xgb.best_iteration,
        },
        os.path.join(MODEL_DIR, "model_metadata.joblib"),
    )
    print(f"\n✅ Modèle + métadonnées sauvegardés dans {MODEL_DIR}/")
    print(f"   AUC-ROC : {xgb_metrics_f1['auc_roc']:.4f} | AUC-PR : {xgb_metrics_f1['auc_pr']:.4f}")
    print(f"   Seuil F1-optimal   : {xgb_metrics_f1['threshold']:.3f}")
    print(f"   Seuil recall>=60%  : {xgb_metrics_recall['threshold']:.3f}")


if __name__ == "__main__":
    main()