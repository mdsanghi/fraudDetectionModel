"""
Fraud Detection using XGBoost
=============================
Reads fraud detection data from a multi-sheet Excel file,
merges Customers, Beneficiaries, Transactions & Login_History,
engineers features, trains an XGBoost classifier, evaluates
performance, and saves the trained model + visual report.

Usage (from project root):
    python core\ model/xGBoost_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score
)
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── PATHS ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "fraud_detection_synthetic_data.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. DATA LOADING ─────────────────────────────────────────────────

def load_all_sheets(path: str) -> dict:
    sheets_to_read = ["Customers", "Beneficiaries", "Transactions", "Login_History"]
    all_sheets = {s: pd.read_excel(path, sheet_name=s) for s in sheets_to_read}
    for name, df in all_sheets.items():
        print(f"  {name:20s} → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return all_sheets


# ── 2. FEATURE ENGINEERING FROM LOGIN_HISTORY ────────────────────────

def build_login_features(df_login: pd.DataFrame) -> pd.DataFrame:
    """Aggregate login history per customer into useful features."""
    agg = df_login.groupby("customer_id").agg(
        total_logins=("login_id", "count"),
        failed_logins=("login_status", lambda x: (x != "success").sum()),
        avg_session_mins=("session_duration_mins", "mean"),
        new_device_logins=("is_new_device", "sum"),
        new_location_logins=("is_new_location", "sum"),
        mfa_usage_rate=("mfa_used", "mean"),
        unique_ips=("ip_address", "nunique"),
        unique_devices=("device", "nunique"),
        unique_countries=("location_country", "nunique"),
    ).reset_index()
    agg["failed_login_rate"] = agg["failed_logins"] / agg["total_logins"]
    return agg


# ── 3. FEATURE ENGINEERING FROM BENEFICIARIES ────────────────────────

def build_beneficiary_features(df_ben: pd.DataFrame) -> pd.DataFrame:
    """Aggregate beneficiary info per customer."""
    agg = df_ben.groupby("customer_id").agg(
        total_beneficiaries=("beneficiary_id", "count"),
        verified_beneficiaries=("is_verified", "sum"),
        unique_ben_countries=("country", "nunique"),
        unique_ben_banks=("bank_name", "nunique"),
    ).reset_index()
    agg["unverified_ratio"] = 1 - (agg["verified_beneficiaries"] / agg["total_beneficiaries"])
    return agg


# ── 4. MERGE ALL DATA ───────────────────────────────────────────────

def merge_data(all_sheets: dict) -> pd.DataFrame:
    df_txn = all_sheets["Transactions"].copy()
    df_cust = all_sheets["Customers"].copy()
    df_ben = all_sheets["Beneficiaries"].copy()
    df_login = all_sheets["Login_History"].copy()

    # Build aggregated features
    login_features = build_login_features(df_login)
    ben_features = build_beneficiary_features(df_ben)

    # Merge: Transactions ← Customers (on customer_id)
    df = df_txn.merge(df_cust, on="customer_id", how="left", suffixes=("", "_cust"))

    # Merge: ← Login features (on customer_id)
    df = df.merge(login_features, on="customer_id", how="left")

    # Merge: ← Beneficiary features (on customer_id)
    df = df.merge(ben_features, on="customer_id", how="left")

    print(f"\n  Merged dataset → {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── 5. PREPROCESSING ────────────────────────────────────────────────

TARGET_COL = "label"

# Columns to drop: IDs, names, free-text, redundant nationality names
DROP_COLS = [
    "transaction_id", "customer_id", "beneficiary_id",
    "full_name", "email", "phone", "account_number",
    "beneficiary_name", "ip_address",
    "beneficiary_nationality_name", "customer_nationality",
    "nationality_name", "nationality_name_cust",
    "fraud_type", "suspicious_flags", "status",
]


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Drop non-predictive columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Separate target
    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])

    # Encode target if needed
    le_target = None
    if y.dtype == "object":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=TARGET_COL)
        print(f"  Target mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # Handle datetime columns → extract numeric parts then drop
    for col in X.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X[f"{col}_hour"] = X[col].dt.hour
        X[f"{col}_dayofweek"] = X[col].dt.dayofweek
        X.drop(columns=[col], inplace=True)

    # Encode remaining categorical columns
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Fill nulls
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(X[col].median() if X[col].dtype in ("float64", "int64") else 0, inplace=True)

    print(f"  Final features: {X.shape[1]}  |  Samples: {X.shape[0]:,}")
    print(f"  Target distribution: {dict(y.value_counts())}")
    return X, y, label_encoders, le_target


# ── 6. MODEL TRAINING ───────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val):
    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  scale_pos_weight: {fraud_ratio:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_ratio,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    return model


# ── 7. EVALUATION ────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, feature_names, output_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision": round(average_precision_score(y_test, y_proba), 4),
    }
    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  PR-AUC  : {metrics['avg_precision']}")

    # ── Dashboard ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Fraud Detection — XGBoost Evaluation", fontsize=16, fontweight="bold", y=0.98)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("Actual")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 1].set_title("ROC Curve"); axes[0, 1].set_xlabel("FPR"); axes[0, 1].set_ylabel("TPR")
    axes[0, 1].legend()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[1, 0].plot(rec, prec, lw=2, color="#ff7f0e", label=f"AP = {metrics['avg_precision']:.3f}")
    axes[1, 0].set_title("Precision-Recall Curve"); axes[1, 0].set_xlabel("Recall"); axes[1, 0].set_ylabel("Precision")
    axes[1, 0].legend()

    # Feature importance (top 15)
    importance = model.feature_importances_
    top_n = min(15, len(feature_names))
    idx = np.argsort(importance)[-top_n:]
    axes[1, 1].barh(range(top_n), importance[idx], color="#2ca02c")
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([feature_names[i] for i in idx])
    axes[1, 1].set_title(f"Top {top_n} Feature Importances")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    chart_path = os.path.join(output_dir, "evaluation_dashboard.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dashboard saved → {chart_path}")
    return metrics


# ── 8. CROSS VALIDATION ─────────────────────────────────────────────

def cross_validate_model(model, X, y):
    cv_model = xgb.XGBClassifier(
        n_estimators=model.best_iteration + 1 if hasattr(model, "best_iteration") else 200,
        max_depth=model.get_params()["max_depth"],
        learning_rate=model.get_params()["learning_rate"],
        subsample=model.get_params()["subsample"],
        colsample_bytree=model.get_params()["colsample_bytree"],
        scale_pos_weight=model.get_params()["scale_pos_weight"],
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(cv_model, X, y, cv=cv, scoring="roc_auc")
    print(f"\n── 5-Fold CV ROC-AUC ──")
    print(f"  Scores: {np.round(scores, 4)}")
    print(f"  Mean  : {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


# ── 9. MAIN PIPELINE ────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FRAUD DETECTION SYSTEM — XGBoost Training Pipeline")
    print("=" * 60)

    # Load
    print(f"\n── Loading data from: {DATA_PATH} ──")
    all_sheets = load_all_sheets(DATA_PATH)

    # Merge
    print("\n── Merging tables ──")
    df = merge_data(all_sheets)

    # Preprocess
    print("\n── Preprocessing ──")
    X, y, label_encoders, le_target = preprocess(df)

    # Split: 80% train+val / 20% test, then 85% train / 15% val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # Train
    print("\n── Training XGBoost ──")
    model = train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n── Evaluation ──")
    metrics = evaluate(model, X_test, y_test, list(X.columns), OUTPUT_DIR)

    # Cross-validate
    cv_scores = cross_validate_model(model, X, y)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "xgb_fraud_model.json")
    model.save_model(model_path)
    print(f"\n✓ Model saved    → {model_path}")

    # Save metadata
    meta = {
        "features": list(X.columns),
        "target": TARGET_COL,
        "metrics": metrics,
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "label_encoders": {col: list(le.classes_) for col, le in label_encoders.items()},
    }
    if le_target:
        meta["target_mapping"] = dict(zip(le_target.classes_.tolist(), range(len(le_target.classes_))))
    meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"✓ Metadata saved → {meta_path}")

    print("\n" + "=" * 60)
    print("  DONE — Model ready for deployment")
    print("=" * 60)


if __name__ == "__main__":
    main()