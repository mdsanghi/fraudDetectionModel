"""
Fraud Detection System — XGBoost Pipeline
==========================================
Complete ML pipeline: Data loading → EDA → Feature engineering →
Stratified split → Hyperparameter tuning → Training → Evaluation
(F1, Precision, Recall, Accuracy, ROC-AUC, PR-AUC) → Charts & graphs.

Usage (from project root):
    python "core model/xGBoost_model.py"
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
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
import xgboost as xgb
import joblib
from scipy.stats import uniform, randint
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score,
    roc_auc_score, brier_score_loss
)
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)
SEARCH_N_JOBS = 1

# ═══════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════
# When running from "core model/" subfolder, BASE_DIR goes up to project root.
# For direct execution, fall back to current working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_DATA_IN_ROOT = os.path.join(_PROJECT_ROOT, "fraud_detection_synthetic_data.xlsx")
_DATA_IN_CWD = os.path.join(os.getcwd(), "fraud_detection_synthetic_data.xlsx")

if os.path.exists(_DATA_IN_ROOT):
    BASE_DIR = _PROJECT_ROOT
elif os.path.exists(_DATA_IN_CWD):
    BASE_DIR = os.getcwd()
else:
    BASE_DIR = _PROJECT_ROOT  # default

DATA_PATH = os.path.join(BASE_DIR, "fraud_detection_synthetic_data.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

TARGET_COL = "label"
DATETIME_HINTS = ("date", "timestamp", "dob", "birth")

# Columns to drop: IDs, names, free-text, data-leakage
DROP_COLS = [
    "transaction_id", "customer_id", "beneficiary_id",
    "full_name", "email", "phone", "account_number",
    "beneficiary_name", "ip_address",
    "beneficiary_nationality_name", "customer_nationality",
    "nationality_name", "nationality_name_cust",
    "fraud_type", "suspicious_flags", "status",
]


# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def load_all_sheets(path):
    print("=" * 70)
    print("  STEP 1: DATA LOADING")
    print("=" * 70)
    sheets = ["Customers", "Beneficiaries", "Transactions", "Login_History"]
    data = {s: pd.read_excel(path, sheet_name=s) for s in sheets}
    for name, df in data.items():
        print(f"  {name:20s} → {df.shape[0]:>5,} rows × {df.shape[1]:>2} columns")
    return data


# ═══════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ═══════════════════════════════════════════════════════════════════
def run_eda(all_sheets, charts_dir):
    print("\n" + "=" * 70)
    print("  STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    df_txn = all_sheets["Transactions"]
    df_cust = all_sheets["Customers"]
    df_ben = all_sheets["Beneficiaries"]
    df_login = all_sheets["Login_History"]

    # --- 2a. Target distribution ---
    print("\n  ── Target distribution (Transactions) ──")
    target_counts = df_txn[TARGET_COL].value_counts()
    print(f"  {dict(target_counts)}")
    print(f"  Fraud rate: {(target_counts.get(1, 0) / len(df_txn) * 100):.2f}%")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Exploratory Data Analysis — Fraud Detection", fontsize=18, fontweight="bold", y=1.02)

    # Chart 1: Target distribution
    colors = ["#2ca02c", "#d62728"]
    target_counts.plot(kind="bar", ax=axes[0, 0], color=colors, edgecolor="white")
    axes[0, 0].set_title("Target distribution (label)")
    axes[0, 0].set_xlabel("Label (0=Legit, 1=Fraud)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].tick_params(axis="x", rotation=0)
    for i, v in enumerate(target_counts.values):
        axes[0, 0].text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")

    # Chart 2: Transaction amount by label
    print("\n  ── Transaction amount stats by label ──")
    print(df_txn.groupby(TARGET_COL)["amount_usd"].describe().round(2).to_string())

    df_txn.boxplot(column="amount_usd", by=TARGET_COL, ax=axes[0, 1],
                   boxprops=dict(color="#1f77b4"), medianprops=dict(color="#d62728"))
    axes[0, 1].set_title("Amount (USD) by label")
    axes[0, 1].set_xlabel("Label (0=Legit, 1=Fraud)")
    axes[0, 1].set_ylabel("Amount USD")
    plt.sca(axes[0, 1])
    plt.title("Amount (USD) by label")

    # Chart 3: Transaction hour distribution
    axes[0, 2].hist(df_txn[df_txn[TARGET_COL] == 0]["transaction_hour"],
                     bins=24, alpha=0.6, label="Legit", color="#2ca02c", edgecolor="white")
    axes[0, 2].hist(df_txn[df_txn[TARGET_COL] == 1]["transaction_hour"],
                     bins=24, alpha=0.7, label="Fraud", color="#d62728", edgecolor="white")
    axes[0, 2].set_title("Transaction hour distribution")
    axes[0, 2].set_xlabel("Hour of day")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].legend()

    # Chart 4: Risk tier distribution
    if "risk_tier" in df_cust.columns:
        risk_counts = df_cust["risk_tier"].value_counts()
        risk_counts.plot(kind="bar", ax=axes[1, 0], color="#ff7f0e", edgecolor="white")
        axes[1, 0].set_title("Customer risk tier distribution")
        axes[1, 0].set_xlabel("Risk Tier")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].tick_params(axis="x", rotation=45)

    # Chart 5: Cross-nationality vs fraud
    if "is_cross_nationality" in df_txn.columns:
        cross_nat = df_txn.groupby(["is_cross_nationality", TARGET_COL]).size().unstack(fill_value=0)
        cross_nat.plot(kind="bar", ax=axes[1, 1], color=colors, edgecolor="white")
        axes[1, 1].set_title("Cross-nationality vs fraud")
        axes[1, 1].set_xlabel("Is cross-nationality")
        axes[1, 1].legend(["Legit", "Fraud"])
        axes[1, 1].tick_params(axis="x", rotation=0)

    # Chart 6: High risk country vs fraud
    if "is_high_risk_country" in df_txn.columns:
        hrc = df_txn.groupby(["is_high_risk_country", TARGET_COL]).size().unstack(fill_value=0)
        hrc.plot(kind="bar", ax=axes[1, 2], color=colors, edgecolor="white")
        axes[1, 2].set_title("High-risk country vs fraud")
        axes[1, 2].set_xlabel("Is high-risk country")
        axes[1, 2].legend(["Legit", "Fraud"])
        axes[1, 2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    eda_path = os.path.join(charts_dir, "01_eda_overview.png")
    fig.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ EDA chart saved → {eda_path}")

    # --- 2b. Missing values report ---
    print("\n  ── Missing values per sheet ──")
    for name, df in all_sheets.items():
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if len(nulls) > 0:
            print(f"  {name}: {dict(nulls)}")
        else:
            print(f"  {name}: No missing values")

    # --- 2c. Correlation heatmap for numeric transaction features ---
    numeric_cols = df_txn.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 2:
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        corr = df_txn[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=ax2, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        ax2.set_title("Transaction features — correlation matrix", fontsize=14, fontweight="bold")
        corr_path = os.path.join(charts_dir, "02_correlation_matrix.png")
        fig2.savefig(corr_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  ✓ Correlation matrix saved → {corr_path}")

    return df_txn


# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
def build_login_features(df_login):
    """Aggregate login history per customer — captures repeated attempts & location shifts."""
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

    # ── RULE: Repeated login attempts ──
    agg["failed_login_rate"] = agg["failed_logins"] / agg["total_logins"]
    agg["has_repeated_failures"] = (agg["failed_logins"] >= 3).astype(int)

    # ── RULE: Recent login location shift ──
    # Compare last login country to most-common country per customer
    last_login = (
        df_login.sort_values("login_timestamp")
        .groupby("customer_id")
        .agg(last_login_country=("location_country", "last"))
        .reset_index()
    )
    mode_country = (
        df_login.groupby("customer_id")["location_country"]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        .reset_index()
        .rename(columns={"location_country": "usual_login_country"})
    )
    agg = agg.merge(last_login, on="customer_id", how="left")
    agg = agg.merge(mode_country, on="customer_id", how="left")
    agg["login_location_shift"] = (
        agg["last_login_country"] != agg["usual_login_country"]
    ).astype(int)
    agg.drop(columns=["last_login_country", "usual_login_country"], inplace=True)

    # ── RULE: Low MFA usage (no MFA habit) ──
    agg["low_mfa_user"] = (agg["mfa_usage_rate"] < 0.3).astype(int)

    return agg


def build_beneficiary_features(df_ben):
    """Aggregate beneficiary info — captures verification & recency signals."""
    df_ben = df_ben.copy()
    df_ben["added_date"] = pd.to_datetime(df_ben["added_date"], errors="coerce")

    agg = df_ben.groupby("customer_id").agg(
        total_beneficiaries=("beneficiary_id", "count"),
        verified_beneficiaries=("is_verified", "sum"),
        unique_ben_countries=("country", "nunique"),
        unique_ben_banks=("bank_name", "nunique"),
    ).reset_index()

    # ── RULE: Beneficiary verification ratio ──
    agg["unverified_ratio"] = 1 - (agg["verified_beneficiaries"] / agg["total_beneficiaries"])
    agg["has_unverified_ben"] = (agg["unverified_ratio"] > 0).astype(int)

    # ── RULE: Recently added beneficiary (within last 7 days) ──
    now = pd.Timestamp.now()
    recent = df_ben.copy()
    recent["days_since_added"] = (now - recent["added_date"]).dt.days
    recent_agg = recent.groupby("customer_id").agg(
        newest_ben_days=("days_since_added", "min"),
        ben_added_last_7d=("days_since_added", lambda x: (x <= 7).sum()),
        ben_added_last_30d=("days_since_added", lambda x: (x <= 30).sum()),
    ).reset_index()
    agg = agg.merge(recent_agg, on="customer_id", how="left")
    agg["has_very_recent_ben"] = (agg["newest_ben_days"] <= 7).astype(int)

    return agg


def merge_and_engineer(all_sheets):
    print("\n" + "=" * 70)
    print("  STEP 3: FEATURE ENGINEERING & MERGE")
    print("=" * 70)

    df_txn = all_sheets["Transactions"].copy()
    df_cust = all_sheets["Customers"].copy()

    # Customer account age
    if "registration_date" in df_cust.columns:
        df_cust["registration_date"] = pd.to_datetime(df_cust["registration_date"], errors="coerce")
        df_cust["account_age_days"] = (pd.Timestamp.now() - df_cust["registration_date"]).dt.days

    login_feat = build_login_features(all_sheets["Login_History"])
    ben_feat = build_beneficiary_features(all_sheets["Beneficiaries"])

    # Merge all onto transactions
    df = df_txn.merge(df_cust, on="customer_id", how="left", suffixes=("", "_cust"))
    df = df.merge(login_feat, on="customer_id", how="left")
    df = df.merge(ben_feat, on="customer_id", how="left")

    # ═════════════════════════════════════════════════════════════
    # FRAUD RULE FEATURES — explicitly encoded for the model
    # ═════════════════════════════════════════════════════════════

    # ── RULE 1: Amount spike (2x, 3x, 5x over customer average) ──
    if "amount_usd" in df.columns and "customer_avg_amount" in df.columns:
        avg = df["customer_avg_amount"].replace(0, np.nan)
        df["amount_ratio_to_avg"] = df["amount_usd"] / avg
        df["amount_ratio_to_avg"] = df["amount_ratio_to_avg"].fillna(1.0)
        df["spike_2x"] = (df["amount_ratio_to_avg"] >= 2.0).astype(int)
        df["spike_3x"] = (df["amount_ratio_to_avg"] >= 3.0).astype(int)
        df["spike_5x"] = (df["amount_ratio_to_avg"] >= 5.0).astype(int)

    if "amount_usd" in df.columns and "typical_txn_amount" in df.columns:
        typ = df["typical_txn_amount"].replace(0, np.nan)
        df["amount_vs_typical_ratio"] = (df["amount_usd"] / typ).fillna(1.0)
        df["spike_typical_2x"] = (df["amount_vs_typical_ratio"] >= 2.0).astype(int)
        df["spike_typical_5x"] = (df["amount_vs_typical_ratio"] >= 5.0).astype(int)

    # ── RULE 2: New device ──
    #   is_new_device already in transactions
    if "device_used" in df.columns and "primary_device" in df.columns:
        df["device_mismatch"] = (df["device_used"] != df["primary_device"]).astype(int)

    # ── RULE 3: New beneficiary ──
    #   is_new_beneficiary already in transactions

    # ── RULE 4: Cross-nationality + new beneficiary (compound rule) ──
    if "is_cross_nationality" in df.columns and "is_new_beneficiary" in df.columns:
        df["cross_nat_new_ben"] = (
            (df["is_cross_nationality"] == 1) & (df["is_new_beneficiary"] == 1)
        ).astype(int)

    # ── RULE 5: High-risk country ──
    #   is_high_risk_country already in transactions

    # ── RULE 6: Unusual hour / deep night (00:00–05:59) ──
    if "transaction_hour" in df.columns:
        df["is_deep_night"] = df["transaction_hour"].between(0, 5).astype(int)
    #   is_unusual_hour already in transactions

    # ── RULE 7: Repeated login attempts ──
    #   login_attempts_before already in transactions
    if "login_attempts_before" in df.columns:
        df["high_login_attempts"] = (df["login_attempts_before"] >= 3).astype(int)
    #   has_repeated_failures, failed_login_rate from login_feat

    # ── RULE 8: No MFA for high-value transfer ──
    if "mfa_usage_rate" in df.columns and "amount_usd" in df.columns:
        median_amount = df["amount_usd"].median()
        df["high_value_txn"] = (df["amount_usd"] > median_amount * 2).astype(int)
        df["no_mfa_high_value"] = (
            (df["low_mfa_user"] == 1) & (df["high_value_txn"] == 1)
        ).astype(int)

    # ── RULE 9: Customer risk tier / KYC / account status ──
    #   risk_tier, kyc_status, account_status from customers (already merged)
    if "risk_tier" in df.columns:
        df["is_high_risk_customer"] = (df["risk_tier"] == "high").astype(int)
    if "kyc_status" in df.columns:
        df["kyc_not_verified"] = (df["kyc_status"] != "verified").astype(int)
    if "account_status" in df.columns:
        df["account_not_active"] = (df["account_status"] != "active").astype(int)

    # ── RULE 10: Beneficiary verification / recently added ──
    #   has_unverified_ben, has_very_recent_ben, unverified_ratio from ben_feat
    if "is_new_beneficiary" in df.columns and "has_unverified_ben" in df.columns:
        df["new_unverified_ben"] = (
            (df["is_new_beneficiary"] == 1) & (df["has_unverified_ben"] == 1)
        ).astype(int)

    # ── RULE 11: Recent login location shift ──
    #   login_location_shift from login_feat (already merged)

    # ── Channel mismatch ──
    if "channel" in df.columns and "preferred_channel" in df.columns:
        df["channel_mismatch"] = (df["channel"] != df["preferred_channel"]).astype(int)

    # ── COMPOUND: Multi-signal risk score (count how many rules fire) ──
    rule_flags = [
        "spike_2x", "is_new_device", "is_new_beneficiary", "cross_nat_new_ben",
        "is_high_risk_country", "is_deep_night", "high_login_attempts",
        "no_mfa_high_value", "is_high_risk_customer", "kyc_not_verified",
        "new_unverified_ben", "login_location_shift", "device_mismatch"
    ]
    existing_flags = [f for f in rule_flags if f in df.columns]
    df["fraud_signal_count"] = df[existing_flags].sum(axis=1)
    df["multi_signal_alert"] = (df["fraud_signal_count"] >= 3).astype(int)

    engineered = [c for c in df.columns if c not in all_sheets["Transactions"].columns
                  and c not in all_sheets["Customers"].columns and c != TARGET_COL]
    print(f"  Merged dataset       → {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Engineered features  → {len(engineered)}")
    print(f"  Fraud rule features  :")
    rule_features = [
        ("Amount spike 2x/3x/5x", ["spike_2x", "spike_3x", "spike_5x", "spike_typical_2x", "spike_typical_5x"]),
        ("New device",             ["is_new_device", "device_mismatch"]),
        ("New beneficiary",        ["is_new_beneficiary", "cross_nat_new_ben"]),
        ("High-risk country",      ["is_high_risk_country"]),
        ("Unusual hour / night",   ["is_unusual_hour", "is_deep_night"]),
        ("Repeated login attempts",["high_login_attempts", "has_repeated_failures", "failed_login_rate"]),
        ("No MFA + high value",    ["no_mfa_high_value", "low_mfa_user", "high_value_txn"]),
        ("Risk tier / KYC",        ["is_high_risk_customer", "kyc_not_verified", "account_not_active"]),
        ("Beneficiary risk",       ["has_unverified_ben", "has_very_recent_ben", "new_unverified_ben"]),
        ("Login location shift",   ["login_location_shift"]),
        ("Compound signal",        ["fraud_signal_count", "multi_signal_alert"]),
    ]
    for rule_name, cols in rule_features:
        present = [c for c in cols if c in df.columns]
        print(f"    ✓ {rule_name:28s} → {present}")

    return df


def coerce_datetime_like_columns(df):
    """Convert parseable date-like object columns to datetime for stable features."""
    df = df.copy()
    converted = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            converted.append(col)
            continue
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        if not any(hint in col.lower() for hint in DATETIME_HINTS):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.6:
            df[col] = parsed
            converted.append(col)
    return df, converted


# ═══════════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
def preprocess(df):
    print("\n" + "=" * 70)
    print("  STEP 4: PREPROCESSING")
    print("=" * 70)

    df = df.copy()
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped {len(cols_to_drop)} non-predictive / leakage columns")

    y = df[TARGET_COL].copy()
    X = df.drop(columns=[TARGET_COL])

    # Encode target
    le_target = None
    if y.dtype == "object":
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=TARGET_COL)
        print(f"  Target mapping: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    # Convert parseable date-like strings before encoding.
    X, converted_datetime_cols = coerce_datetime_like_columns(X)
    if converted_datetime_cols:
        print(f"  Parsed datetime-like columns: {converted_datetime_cols}")

    # Preserve numeric semantics for count/rate fields that may have landed as objects.
    for col in X.select_dtypes(include=["object", "category"]).columns:
        numeric_candidate = pd.to_numeric(X[col], errors="coerce")
        if numeric_candidate.notna().mean() >= 0.98:
            X[col] = numeric_candidate

    # Datetime → numeric components
    for col in X.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X[f"{col}_hour"] = X[col].dt.hour
        X[f"{col}_dayofweek"] = X[col].dt.dayofweek
        X.drop(columns=[col], inplace=True)

    # Label encode categoricals
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Replace infinities and fill nulls (use assignment, not inplace, for pandas CoW)
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isnull().any():
            fill_val = X[col].median() if X[col].dtype in ("float64", "float32", "int64", "int32") else 0
            if pd.isna(fill_val):
                fill_val = 0
            X[col] = X[col].fillna(fill_val)

    # Drop zero-variance columns (useless for model, cause NaN in StandardScaler)
    zero_var_cols = [col for col in X.columns if X[col].std() == 0]
    if zero_var_cols:
        X.drop(columns=zero_var_cols, inplace=True)
        print(f"  Dropped {len(zero_var_cols)} zero-variance columns: {zero_var_cols}")

    print(f"  Final features : {X.shape[1]}")
    print(f"  Total samples  : {X.shape[0]:,}")
    print(f"  Target balance : {dict(y.value_counts())}")
    return X, y, label_encoders, le_target


# ═══════════════════════════════════════════════════════════════════
# 5. STRATIFIED SPLIT
# ═══════════════════════════════════════════════════════════════════
def stratified_split(X, y):
    print("\n" + "=" * 70)
    print("  STEP 5: STRATIFIED TRAIN/TEST SPLIT (80/20)")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("  STEP 6: STRATIFIED TRAIN/TEST SPLIT (80/20)")
    print("=" * 70)

    # Use index-based splitting
    indices = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=42
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15, stratify=y.iloc[idx_trainval], random_state=42
    )

    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y.iloc[idx_train], y.iloc[idx_val], y.iloc[idx_test]

    print(f"  Train : {X_train.shape[0]:>5,} samples ({y_train.mean():.2%} fraud)")
    print(f"  Val   : {X_val.shape[0]:>5,} samples ({y_val.mean():.2%} fraud)")
    print(f"  Test  : {X_test.shape[0]:>5,} samples ({y_test.mean():.2%} fraud)")
    print(f"  ✓ Stratification preserved across all splits")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════════
# 7. HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════════════
def tune_hyperparameters(X_train, y_train):
    print("\n" + "=" * 70)
    print("  STEP 7: HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 70)

    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.5, 0.5),
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 0.5),
        "reg_alpha": uniform(0, 1.0),
        "reg_lambda": uniform(0.5, 1.5),
        "scale_pos_weight": [fraud_ratio],
    }

    base_model = xgb.XGBClassifier(
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1",
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=SEARCH_N_JOBS,
        return_train_score=True,
    )

    print("  Running 30 iterations × 5-fold CV = 150 fits...")
    search.fit(X_train, y_train)

    best = search.best_params_
    print(f"\n  ── Best hyperparameters ──")
    for k, v in sorted(best.items()):
        print(f"    {k:25s}: {v}")
    print(f"  Best CV F1: {search.best_score_:.4f}")

    return search.best_params_, search


# ═══════════════════════════════════════════════════════════════════
# 8. MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════
def train_model(X_train, y_train, X_val, y_val, best_params):
    print("\n" + "=" * 70)
    print("  STEP 8: TRAINING FINAL MODEL")
    print("=" * 70)

    model = xgb.XGBClassifier(
        **best_params,
        eval_metric=["error", "logloss", "aucpr"],
        early_stopping_rounds=30,
        random_state=42,
        tree_method="hist",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"  ✓ Best iteration: {best_iter}")
    return model, model.evals_result()


def compute_iteration_classification_metrics(model, X_train, y_train, X_val, y_val):
    """
    Capture classification quality after each boosting round.
    This lets us inspect both the score at each iteration and the gain
    compared with the previous iteration.
    """
    print("  Computing per-iteration accuracy and precision...")
    total_rounds = len(model.evals_result().get("validation_0", {}).get("logloss", []))
    if total_rounds == 0:
        return pd.DataFrame()

    rows = []
    for iteration in range(1, total_rounds + 1):
        train_proba = model.predict_proba(X_train, iteration_range=(0, iteration))[:, 1]
        val_proba = model.predict_proba(X_val, iteration_range=(0, iteration))[:, 1]
        train_pred = (train_proba >= 0.5).astype(int)
        val_pred = (val_proba >= 0.5).astype(int)

        rows.append({
            "iteration": iteration,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "train_precision": precision_score(y_train, train_pred, zero_division=0),
            "val_precision": precision_score(y_val, val_pred, zero_division=0),
        })

    metrics_df = pd.DataFrame(rows)
    for col in ["train_accuracy", "val_accuracy", "train_precision", "val_precision"]:
        metrics_df[f"{col}_increase"] = metrics_df[col].diff().fillna(metrics_df[col])

    return metrics_df



# ═══════════════════════════════════════════════════════════════════
# PLOT A — Learning curves (loss, error, aucpr per iteration)
# ═══════════════════════════════════════════════════════════════════
def plot_learning_curves(evals_result, charts_dir):
    """Plot train vs validation metrics across every boosting iteration."""
    print("  Plotting learning curves...")
    train_r = evals_result.get("validation_0", {})
    val_r   = evals_result.get("validation_1", {})

    metrics_to_plot = [k for k in train_r if k in val_r]
    n = len(metrics_to_plot)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Learning Curves — Train vs Validation per Iteration",
                 fontsize=15, fontweight="bold")

    metric_labels = {"error": "Classification Error", "logloss": "Log Loss", "aucpr": "PR-AUC"}
    for ax, metric in zip(axes, metrics_to_plot):
        iters = range(1, len(train_r[metric]) + 1)
        ax.plot(iters, train_r[metric], lw=2, color="#1f77b4", label="Train")
        ax.plot(iters, val_r[metric],   lw=2, color="#d62728", label="Validation", linestyle="--")
        # Mark best iteration
        if metric == "aucpr":
            best_i = int(np.argmax(val_r[metric]))
        else:
            best_i = int(np.argmin(val_r[metric]))
        ax.axvline(x=best_i+1, color="black", linestyle=":", alpha=0.6, lw=1.5)
        ax.scatter(best_i+1, val_r[metric][best_i], s=80, zorder=5,
                   color="black", label=f"Best iter={best_i+1}")
        ax.set_title(metric_labels.get(metric, metric), fontweight="bold")
        ax.set_xlabel("Boosting Iteration")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(charts_dir, "03a_learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Learning curves saved → {path}")


def plot_iteration_metric_growth(iteration_metrics, charts_dir):
    """Plot accuracy and precision trends plus their per-iteration increases."""
    if iteration_metrics.empty:
        return

    print("  Plotting per-iteration accuracy and precision growth...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    fig.suptitle("Per-Iteration Accuracy and Precision Growth", fontsize=15, fontweight="bold")

    iters = iteration_metrics["iteration"]

    axes[0, 0].plot(iters, iteration_metrics["train_accuracy"], lw=2, color="#1f77b4", label="Train")
    axes[0, 0].plot(iters, iteration_metrics["val_accuracy"], lw=2, color="#d62728",
                    linestyle="--", label="Validation")
    axes[0, 0].set_title("Accuracy by Iteration", fontweight="bold")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, iteration_metrics["train_precision"], lw=2, color="#1f77b4", label="Train")
    axes[0, 1].plot(iters, iteration_metrics["val_precision"], lw=2, color="#d62728",
                    linestyle="--", label="Validation")
    axes[0, 1].set_title("Precision by Iteration", fontweight="bold")
    axes[0, 1].set_ylabel("Precision")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(iters, iteration_metrics["val_accuracy_increase"], color="#2ca02c", alpha=0.8)
    axes[1, 0].axhline(0, color="black", lw=1)
    axes[1, 0].set_title("Validation Accuracy Increase per Iteration", fontweight="bold")
    axes[1, 0].set_xlabel("Boosting Iteration")
    axes[1, 0].set_ylabel("Delta Accuracy")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(iters, iteration_metrics["val_precision_increase"], color="#ff7f0e", alpha=0.8)
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set_title("Validation Precision Increase per Iteration", fontweight="bold")
    axes[1, 1].set_xlabel("Boosting Iteration")
    axes[1, 1].set_ylabel("Delta Precision")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(charts_dir, "03c_iteration_metric_growth.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Iteration metric growth plot saved → {path}")


def save_iteration_metrics(iteration_metrics, output_dir):
    """Save raw per-iteration metrics for inspection in Excel or pandas."""
    if iteration_metrics.empty:
        return

    path = os.path.join(output_dir, "iteration_metrics.csv")
    iteration_metrics.to_csv(path, index=False)
    print(f"  ✓ Iteration metrics saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# PLOT B — Overfitting gap (train-val delta across iterations)
# ═══════════════════════════════════════════════════════════════════
def plot_overfitting_analysis(evals_result, charts_dir):
    """Visualise the gap between train and validation metric per iteration."""
    print("  Plotting overfitting analysis...")
    train_r = evals_result.get("validation_0", {})
    val_r   = evals_result.get("validation_1", {})

    if "logloss" not in train_r:
        return

    iters = range(1, len(train_r["logloss"]) + 1)
    gap   = [t - v for t, v in zip(train_r["logloss"], val_r["logloss"])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Overfitting Analysis", fontsize=14, fontweight="bold")

    # Left: both curves
    axes[0].plot(iters, train_r["logloss"], lw=2, color="#1f77b4", label="Train Loss")
    axes[0].plot(iters, val_r["logloss"],   lw=2, color="#d62728", label="Val Loss", linestyle="--")
    axes[0].fill_between(iters, train_r["logloss"], val_r["logloss"],
                          alpha=0.15, color="#ff7f0e", label="Gap (overfit region)")
    axes[0].set_title("Log Loss — Train vs Validation")
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Log Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Right: gap magnitude
    gap_colors = ["#d62728" if g > 0.02 else "#ff7f0e" if g > 0.005 else "#2ca02c" for g in gap]
    axes[1].bar(iters, gap, color=gap_colors, alpha=0.7, width=1.0)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_title("Train-Val Loss Gap per Iteration\n(green=good, amber=watch, red=overfit)")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Gap (Train − Val Loss)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(charts_dir, "03b_overfitting_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Overfitting analysis saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# PLOT C — Hyperparameter search results
# ═══════════════════════════════════════════════════════════════════
def plot_hyperparameter_search(search, charts_dir):
    """Visualise RandomizedSearchCV results to understand tuning landscape."""
    print("  Plotting hyperparameter search results...")
    results_df = pd.DataFrame(search.cv_results_)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Hyperparameter Search — RandomizedSearchCV (30 iterations × 5-fold)",
                 fontsize=15, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. CV score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_df["mean_test_score"], bins=15, color="#1f77b4",
             edgecolor="white", alpha=0.85)
    ax1.axvline(results_df["mean_test_score"].max(), color="#d62728",
                lw=2, linestyle="--", label=f'Best={results_df["mean_test_score"].max():.4f}')
    ax1.set_title("Distribution of CV F1 Scores", fontweight="bold")
    ax1.set_xlabel("Mean CV F1 Score"); ax1.set_ylabel("Count")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Rank vs score
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_df = results_df.sort_values("rank_test_score")
    ax2.scatter(sorted_df["rank_test_score"], sorted_df["mean_test_score"],
                c=sorted_df["mean_test_score"], cmap="RdYlGn", s=60, alpha=0.85, zorder=3)
    ax2.set_title("Rank vs CV F1 Score", fontweight="bold")
    ax2.set_xlabel("Rank (1 = best)"); ax2.set_ylabel("Mean CV F1")
    ax2.grid(True, alpha=0.3)

    # 3. Train vs Val score (overfitting in search)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(results_df["mean_train_score"], results_df["mean_test_score"],
                c=results_df["mean_test_score"], cmap="RdYlGn", s=60, alpha=0.85)
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax3.set_title("Train vs Validation F1\n(above diagonal = generalising well)", fontweight="bold")
    ax3.set_xlabel("Mean Train F1"); ax3.set_ylabel("Mean Val F1")
    ax3.grid(True, alpha=0.3)

    # 4. n_estimators vs score
    ax4 = fig.add_subplot(gs[1, 0])
    if "param_n_estimators" in results_df:
        ax4.scatter(results_df["param_n_estimators"].astype(float),
                    results_df["mean_test_score"],
                    c=results_df["mean_test_score"], cmap="viridis", s=60, alpha=0.85)
        ax4.set_title("n_estimators vs CV F1", fontweight="bold")
        ax4.set_xlabel("n_estimators"); ax4.set_ylabel("Mean CV F1")
        ax4.grid(True, alpha=0.3)

    # 5. learning_rate vs score
    ax5 = fig.add_subplot(gs[1, 1])
    if "param_learning_rate" in results_df:
        ax5.scatter(results_df["param_learning_rate"].astype(float),
                    results_df["mean_test_score"],
                    c=results_df["mean_test_score"], cmap="viridis", s=60, alpha=0.85)
        ax5.set_title("Learning Rate vs CV F1", fontweight="bold")
        ax5.set_xlabel("learning_rate"); ax5.set_ylabel("Mean CV F1")
        ax5.grid(True, alpha=0.3)

    # 6. max_depth vs score
    ax6 = fig.add_subplot(gs[1, 2])
    if "param_max_depth" in results_df:
        depth_vals = results_df["param_max_depth"].astype(float)
        score_vals = results_df["mean_test_score"]
        depth_groups = results_df.groupby("param_max_depth")["mean_test_score"]
        means = depth_groups.mean()
        ax6.bar(means.index.astype(str), means.values, color="#ff7f0e", alpha=0.8, edgecolor="white")
        ax6.set_title("max_depth vs Mean CV F1", fontweight="bold")
        ax6.set_xlabel("max_depth"); ax6.set_ylabel("Mean CV F1")
        ax6.grid(True, alpha=0.3)

    path = os.path.join(charts_dir, "05_hyperparameter_search.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Hyperparameter search plot saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# PLOT D — Extended evaluation plots
# ═══════════════════════════════════════════════════════════════════
def plot_extended_evaluation(y_test, y_pred, y_proba, model,
                              feature_names, charts_dir):
    """Additional deep-dive evaluation plots."""
    print("  Plotting extended evaluation charts...")

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle("Extended Model Evaluation", fontsize=16, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    # ── 1. Threshold analysis ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions, recalls, f1s, accuracies = [], [], [], []
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))
        accuracies.append(accuracy_score(y_test, preds))
    best_thr_idx = int(np.argmax(f1s))
    ax1.plot(thresholds, precisions, lw=2, label="Precision", color="#1f77b4")
    ax1.plot(thresholds, recalls,    lw=2, label="Recall",    color="#d62728")
    ax1.plot(thresholds, f1s,        lw=2, label="F1",        color="#2ca02c")
    ax1.axvline(thresholds[best_thr_idx], color="black", linestyle="--",
                lw=1.5, label=f"Best F1 thr={thresholds[best_thr_idx]:.2f}")
    ax1.set_title("Precision / Recall / F1 vs Threshold", fontweight="bold")
    ax1.set_xlabel("Decision Threshold"); ax1.set_ylabel("Score")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # ── 2. Calibration curve ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    frac_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    ax2.plot(mean_pred, frac_pos, "s-", lw=2, color="#ff7f0e", label="Model")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax2.fill_between(mean_pred, frac_pos, mean_pred,
                     alpha=0.15, color="#ff7f0e", label="Calibration gap")
    ax2.set_title("Calibration Curve\n(how well probabilities match reality)", fontweight="bold")
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Fraction of Positives")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── 3. Cumulative gains chart ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    order   = np.argsort(y_proba)[::-1]
    y_sorted = np.array(y_test)[order]
    total_pos = y_sorted.sum()
    gains = np.cumsum(y_sorted) / total_pos
    pct_pop = np.arange(1, len(y_sorted)+1) / len(y_sorted)
    ax3.plot(pct_pop, gains, lw=2.5, color="#1f77b4", label="Model")
    ax3.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random baseline")
    ax3.fill_between(pct_pop, gains, pct_pop, alpha=0.15, color="#1f77b4", label="Lift area")
    ax3.set_title("Cumulative Gains Curve", fontweight="bold")
    ax3.set_xlabel("% of Population Scored")
    ax3.set_ylabel("% of Fraud Captured")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── 4. Lift curve ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    lift  = gains / pct_pop
    ax4.plot(pct_pop, lift, lw=2.5, color="#2ca02c")
    ax4.axhline(1.0, color="black", linestyle="--", lw=1.5, alpha=0.6, label="No lift baseline")
    ax4.set_title("Lift Curve\n(model vs random at each decile)", fontweight="bold")
    ax4.set_xlabel("% of Population Scored")
    ax4.set_ylabel("Lift")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # ── 5. Error analysis — FP / FN distribution ──────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    fp_proba = y_proba[(y_test == 0) & (y_pred == 1)]   # False Positives
    fn_proba = y_proba[(y_test == 1) & (y_pred == 0)]   # False Negatives
    tp_proba = y_proba[(y_test == 1) & (y_pred == 1)]   # True Positives
    tn_proba = y_proba[(y_test == 0) & (y_pred == 0)]   # True Negatives

    ax5.hist(tn_proba, bins=30, alpha=0.6, color="#2ca02c",   label=f"TN ({len(tn_proba)})", density=True)
    ax5.hist(tp_proba, bins=30, alpha=0.6, color="#1f77b4",   label=f"TP ({len(tp_proba)})", density=True)
    ax5.hist(fp_proba, bins=30, alpha=0.8, color="#ff7f0e",   label=f"FP ({len(fp_proba)})", density=True)
    ax5.hist(fn_proba, bins=30, alpha=0.8, color="#d62728",   label=f"FN ({len(fn_proba)})", density=True)
    ax5.axvline(0.5, color="black", lw=1.5, linestyle="--")
    ax5.set_title("Score Distribution by Prediction Outcome\n(FP/FN show where model struggles)", fontweight="bold")
    ax5.set_xlabel("Predicted Probability"); ax5.set_ylabel("Density")
    ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

    # ── 6. Top-20 feature importance (weight type) ────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    booster   = model.get_booster()
    imp_score = booster.get_score(importance_type="gain")
    imp_df    = pd.DataFrame({"feature": list(imp_score.keys()),
                               "importance": list(imp_score.values())})
    imp_df    = imp_df.nlargest(20, "importance")
    colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
    ax6.barh(imp_df["feature"], imp_df["importance"], color=colors_fi, edgecolor="white")
    ax6.set_title("Top 20 Feature Importances (Gain)", fontweight="bold")
    ax6.set_xlabel("Gain"); ax6.grid(True, alpha=0.3)

    # ── 7. Feature importance — cover ─────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    imp_cover = booster.get_score(importance_type="cover")
    imp_cdf   = pd.DataFrame({"feature": list(imp_cover.keys()),
                               "cover": list(imp_cover.values())})
    imp_cdf   = imp_cdf.nlargest(15, "cover")
    ax7.barh(imp_cdf["feature"], imp_cdf["cover"], color="#ff7f0e", edgecolor="white")
    ax7.set_title("Top 15 Feature Importances (Cover)", fontweight="bold")
    ax7.set_xlabel("Cover"); ax7.grid(True, alpha=0.3)

    # ── 8. Feature importance — frequency ─────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    imp_freq  = booster.get_score(importance_type="weight")
    imp_fdf   = pd.DataFrame({"feature": list(imp_freq.keys()),
                               "weight": list(imp_freq.values())})
    imp_fdf   = imp_fdf.nlargest(15, "weight")
    ax8.barh(imp_fdf["feature"], imp_fdf["weight"], color="#9467bd", edgecolor="white")
    ax8.set_title("Top 15 Feature Importances (Frequency/Weight)", fontweight="bold")
    ax8.set_xlabel("Frequency"); ax8.grid(True, alpha=0.3)

    # ── 9. Extended metrics summary ────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    mcc       = matthews_corrcoef(y_test, y_pred)
    bal_acc   = balanced_accuracy_score(y_test, y_pred)
    brier     = brier_score_loss(y_test, y_proba)
    roc_auc   = roc_auc_score(y_test, y_proba)
    ext_metrics = {
        "MCC": round(mcc, 4),
        "Balanced Acc.": round(bal_acc, 4),
        "Brier Score": round(brier, 4),
        "ROC-AUC": round(roc_auc, 4),
        "F1 @ 0.5": round(f1_score(y_test, y_pred), 4),
        "Precision @ 0.5": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall @ 0.5": round(recall_score(y_test, y_pred, zero_division=0), 4),
    }
    bar_cols = ["#2ca02c" if v >= 0.85 else "#ff7f0e" if v >= 0.6 else "#d62728"
                for v in ext_metrics.values()]
    bars = ax9.barh(list(ext_metrics.keys()), list(ext_metrics.values()),
                    color=bar_cols, edgecolor="white")
    ax9.set_xlim(-0.05, 1.1)
    ax9.set_title("Extended Metrics Summary", fontweight="bold")
    for bar, val in zip(bars, ext_metrics.values()):
        ax9.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    ax9.grid(True, alpha=0.3)

    path = os.path.join(charts_dir, "04b_extended_evaluation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Extended evaluation saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# PLOT E — Cross-validation per-fold detail
# ═══════════════════════════════════════════════════════════════════
def plot_cv_fold_detail(cv_results, charts_dir):
    """Show each fold's individual score to spot instability."""
    print("  Plotting CV fold detail...")
    metrics  = list(cv_results.keys())
    n_folds  = len(cv_results[metrics[0]]["scores"])
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Cross-Validation — Per-Fold Score Detail", fontsize=14, fontweight="bold")

    # Left: grouped bar per metric
    x    = np.arange(n_folds)
    w    = 0.15
    cmap = plt.cm.tab10
    for i, metric in enumerate(metrics):
        scores = cv_results[metric]["scores"]
        axes[0].bar(x + i*w, scores, width=w, label=metric,
                    color=cmap(i/len(metrics)), edgecolor="white", alpha=0.85)
    axes[0].set_xticks(x + w*(len(metrics)-1)/2)
    axes[0].set_xticklabels(fold_labels)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("All Metrics by Fold", fontweight="bold")
    axes[0].set_ylabel("Score"); axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Right: box-plot style spread
    data   = [cv_results[m]["scores"] for m in metrics]
    bp     = axes[1].boxplot(data, labels=metrics, patch_artist=True, notch=False)
    colors = [cmap(i/len(metrics)) for i in range(len(metrics))]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for median_line in bp["medians"]:
        median_line.set_color("black"); median_line.set_linewidth(2)
    axes[1].set_title("Score Spread per Metric (variance = stability)", fontweight="bold")
    axes[1].set_ylabel("Score"); axes[1].set_ylim(0.8, 1.05)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(charts_dir, "07_cv_fold_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ CV fold detail saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# 9. EVALUATION — FULL METRICS
# ═══════════════════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, feature_names, charts_dir, label="Original"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }

    print(f"\n  ── {label} Features — Metrics ──")
    for k, v in metrics.items():
        print(f"    {k:15s}: {v}")

    print(f"\n  ── Classification Report ({label}) ──")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return metrics, y_pred, y_proba


def fit_probability_calibrator(model, X_val, y_val):
    """Fit an isotonic calibrator on validation probabilities and save score scaling."""
    print("\n  Fitting probability calibrator on validation set...")
    raw_val_proba = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val_proba, y_val)
    isotonic_val = calibrator.predict(raw_val_proba)
    blend_alpha = 0.65
    calibrated_val = blend_alpha * raw_val_proba + (1 - blend_alpha) * isotonic_val
    lower_bound = float(np.quantile(calibrated_val, 0.05))
    upper_bound = float(np.quantile(calibrated_val, 0.95))
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 1e-6

    calibration_meta = {
        "method": "isotonic",
        "x_thresholds": [float(v) for v in calibrator.X_thresholds_],
        "y_thresholds": [float(v) for v in calibrator.y_thresholds_],
        "blend_alpha": blend_alpha,
        "ml_score_scale": {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        },
    }
    print(
        "  ✓ Calibration fitted "
        f"(5th pct={lower_bound:.4f}, 95th pct={upper_bound:.4f})"
    )
    return calibration_meta


def apply_calibration(probabilities, calibration_meta):
    """Apply saved isotonic calibration metadata to raw probabilities."""
    if not calibration_meta:
        return probabilities
    if calibration_meta.get("method") != "isotonic":
        return probabilities
    x_thresholds = np.array(calibration_meta.get("x_thresholds", []), dtype=float)
    y_thresholds = np.array(calibration_meta.get("y_thresholds", []), dtype=float)
    if len(x_thresholds) == 0 or len(y_thresholds) == 0:
        return probabilities
    isotonic_probs = np.interp(probabilities, x_thresholds, y_thresholds)
    blend_alpha = float(calibration_meta.get("blend_alpha", 0.65))
    return blend_alpha * probabilities + (1 - blend_alpha) * isotonic_probs


def evaluate_calibrated_model(model, X_test, y_test, feature_names, charts_dir,
                              calibration_meta, label="Calibrated"):
    """Evaluate the model using calibrated probabilities."""
    raw_proba = model.predict_proba(X_test)[:, 1]
    y_proba = apply_calibration(raw_proba, calibration_meta)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
    }

    print(f"\n  ── {label} Features — Metrics ──")
    for k, v in metrics.items():
        print(f"    {k:15s}: {v}")

    print(f"\n  ── Classification Report ({label}) ──")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return metrics, y_pred, y_proba


def plot_evaluation_dashboard(y_test, y_pred, y_proba, metrics, feature_names,
                               model, charts_dir, filename="04_evaluation_dashboard.png"):
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Model Evaluation Dashboard", fontsize=18, fontweight="bold", y=1.02)

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
                annot_kws={"size": 14})
    axes[0, 0].set_title("Confusion matrix", fontweight="bold")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("Actual")

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 1].plot(fpr, tpr, lw=2.5, color="#1f77b4", label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
    axes[0, 1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0, 1].fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")
    axes[0, 1].set_title("ROC curve", fontweight="bold")
    axes[0, 1].set_xlabel("False positive rate")
    axes[0, 1].set_ylabel("True positive rate")
    axes[0, 1].legend(fontsize=12)

    # 3. Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[0, 2].plot(rec, prec, lw=2.5, color="#ff7f0e", label=f"PR-AUC = {metrics['pr_auc']:.4f}")
    axes[0, 2].fill_between(rec, prec, alpha=0.1, color="#ff7f0e")
    axes[0, 2].set_title("Precision-recall curve", fontweight="bold")
    axes[0, 2].set_xlabel("Recall")
    axes[0, 2].set_ylabel("Precision")
    axes[0, 2].legend(fontsize=12)

    # 4. Feature importance (top 15)
    importance = model.feature_importances_
    top_n = min(15, len(feature_names))
    idx = np.argsort(importance)[-top_n:]
    colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    axes[1, 0].barh(range(top_n), importance[idx], color=colors_fi, edgecolor="white")
    axes[1, 0].set_yticks(range(top_n))
    axes[1, 0].set_yticklabels([feature_names[i] for i in idx])
    axes[1, 0].set_title(f"Top {top_n} feature importances", fontweight="bold")
    axes[1, 0].set_xlabel("Importance (gain)")

    # 5. Metrics summary bar chart
    metric_names = list(metrics.keys())
    metric_vals = list(metrics.values())
    bar_colors = ["#2ca02c" if v >= 0.9 else "#ff7f0e" if v >= 0.7 else "#d62728" for v in metric_vals]
    bars = axes[1, 1].barh(metric_names, metric_vals, color=bar_colors, edgecolor="white")
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_title("All metrics summary", fontweight="bold")
    for bar, val in zip(bars, metric_vals):
        axes[1, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontweight="bold")

    # 6. Prediction probability distribution
    axes[1, 2].hist(y_proba[y_test == 0], bins=50, alpha=0.6, label="Legit",
                    color="#2ca02c", edgecolor="white")
    axes[1, 2].hist(y_proba[y_test == 1], bins=50, alpha=0.7, label="Fraud",
                    color="#d62728", edgecolor="white")
    axes[1, 2].set_title("Prediction probability distribution", fontweight="bold")
    axes[1, 2].set_xlabel("Predicted fraud probability")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].legend()
    axes[1, 2].axvline(x=0.5, color="black", linestyle="--", alpha=0.5, label="Threshold=0.5")

    plt.tight_layout()
    path = os.path.join(charts_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Dashboard saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 9. CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════════
def cross_validate_final(best_params, X, y, charts_dir):
    print("\n" + "=" * 70)
    print("  STEP 9: 5-FOLD CROSS VALIDATION")
    print("=" * 70)

    cv_model = xgb.XGBClassifier(
        **best_params,
        eval_metric="aucpr",
        random_state=42,
        tree_method="hist",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    cv_results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(cv_model, X, y, cv=cv, scoring=metric, n_jobs=SEARCH_N_JOBS)
        cv_results[metric] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4), "scores": scores}
        print(f"  {metric:12s}: {scores.mean():.4f} ± {scores.std():.4f}  {np.round(scores, 4)}")

    # CV results chart
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [cv_results[m]["mean"] for m in scoring_metrics]
    stds = [cv_results[m]["std"] for m in scoring_metrics]
    colors = ["#2ca02c" if m >= 0.9 else "#ff7f0e" if m >= 0.7 else "#d62728" for m in means]
    bars = ax.bar(scoring_metrics, means, yerr=stds, capsize=5, color=colors, edgecolor="white", alpha=0.85)
    ax.set_ylim(0, 1.1)
    ax.set_title("5-Fold cross-validation results", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                f"{mean:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    cv_path = os.path.join(charts_dir, "06_cross_validation.png")
    fig.savefig(cv_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ CV chart saved → {cv_path}")

    return cv_results


# ═══════════════════════════════════════════════════════════════════
# 12. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════
def save_artifacts(model, X, best_params, metrics_orig,
                    cv_results, label_encoders, le_target, output_dir,
                    inference_config, calibration_meta):
    print("\n" + "=" * 70)
    print("  STEP 10: SAVING MODEL & ARTIFACTS")
    print("=" * 70)

    # Save model
    model_path = os.path.join(output_dir, "xgb_fraud_model.json")
    model.save_model(model_path)
    print(f"  ✓ Model         → {model_path}")

    # Save model with joblib (for sklearn integration)
    joblib_path = os.path.join(output_dir, "xgb_fraud_model.joblib")
    joblib.dump(model, joblib_path)
    print(f"  ✓ Model (joblib) → {joblib_path}")

    # Save metadata
    meta = {
        "features": list(X.columns),
        "target": TARGET_COL,
        "best_hyperparameters": {k: (v if not isinstance(v, np.integer) else int(v))
                                  for k, v in best_params.items()},
        "metrics": metrics_orig,
        "cross_validation": {k: {"mean": v["mean"], "std": v["std"]}
                              for k, v in cv_results.items()},
        "label_encoders": {col: list(le.classes_) for col, le in label_encoders.items()},
        "inference_config": inference_config,
        "probability_calibration": calibration_meta,
    }
    if le_target:
        meta["target_mapping"] = dict(zip(
            [str(c) for c in le_target.classes_], range(len(le_target.classes_))
        ))

    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  ✓ Metadata       → {meta_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 70)
    print("  FRAUD DETECTION SYSTEM — XGBoost Training Pipeline")
    print("█" * 70)

    # 1. Load
    all_sheets = load_all_sheets(DATA_PATH)

    # 2. EDA
    run_eda(all_sheets, CHARTS_DIR)

    # 3. Feature engineering & merge
    df = merge_and_engineer(all_sheets)

    # 4. Preprocess
    X, y, label_encoders, le_target = preprocess(df)

    # 5. Stratified split
    (X_train, X_val, X_test,
     y_train, y_val, y_test) = stratified_split(X, y)

    # 6. Hyperparameter tuning
    best_params, search = tune_hyperparameters(X_train, y_train)

    # 7. Train final model
    model, evals_result = train_model(X_train, y_train, X_val, y_val, best_params)
    iteration_metrics = compute_iteration_classification_metrics(
        model, X_train, y_train, X_val, y_val
    )
    calibration_meta = fit_probability_calibrator(model, X_val, y_val)

    # 7a. Learning curves (per-iteration train vs val)
    plot_learning_curves(evals_result, CHARTS_DIR)

    # 7b. Accuracy / precision growth per iteration
    plot_iteration_metric_growth(iteration_metrics, CHARTS_DIR)
    save_iteration_metrics(iteration_metrics, OUTPUT_DIR)

    # 7c. Overfitting gap analysis
    plot_overfitting_analysis(evals_result, CHARTS_DIR)

    # 7d. Hyperparameter search landscape
    plot_hyperparameter_search(search, CHARTS_DIR)

    # 8. Evaluate
    metrics_orig, y_pred, y_proba = evaluate_calibrated_model(
        model, X_test, y_test, list(X.columns), CHARTS_DIR, calibration_meta,
        label="Calibrated"
    )
    plot_evaluation_dashboard(y_test, y_pred, y_proba, metrics_orig,
                               list(X.columns), model, CHARTS_DIR)

    # 8b. Extended evaluation (threshold, calibration, lift, FP/FN, 3 importance types)
    plot_extended_evaluation(y_test, y_pred, y_proba, model,
                              list(X.columns), CHARTS_DIR)

    # 9. Cross-validation
    cv_results = cross_validate_final(best_params, X, y, CHARTS_DIR)

    # 9b. CV per-fold detail
    plot_cv_fold_detail(cv_results, CHARTS_DIR)

    inference_config = {
        "high_value_amount_threshold": round(float(df["amount_usd"].median()) * 2, 2),
        "scoring_config": {
            "weights": {"ml": 0.7, "yaml": 0.3},
            "thresholds": {"medium_min": 35, "high_min": 70},
        },
    }

    # 10. Save
    save_artifacts(model, X, best_params, metrics_orig,
                   cv_results, label_encoders, le_target, OUTPUT_DIR,
                   inference_config, calibration_meta)

    # Final summary
    print("\n" + "█" * 70)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("█" * 70)
    print(f"\n  {'Metric':<20} {'Value':>10}")
    print(f"  {'─' * 32}")
    for k in metrics_orig:
        print(f"  {k:<20} {metrics_orig[k]:>10.4f}")
    print(f"\n  Charts saved to: {CHARTS_DIR}")
    print(f"  Model saved to : {OUTPUT_DIR}")
    print("█" * 70)


if __name__ == "__main__":
    main()
