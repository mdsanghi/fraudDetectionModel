"""
Fraud Detection System — XGBoost Pipeline
==========================================
Complete ML pipeline: Data loading → EDA → Feature engineering →
PCA analysis → Stratified split → Hyperparameter tuning →
Training → Evaluation (F1, Precision, Recall, Accuracy, ROC-AUC,
PR-AUC) → Charts & graphs.

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
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score
)
import xgboost as xgb
import joblib
from scipy.stats import uniform, randint

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

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
# 5. PCA ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def run_pca_analysis(X, y, charts_dir):
    print("\n" + "=" * 70)
    print("  STEP 5: PCA ANALYSIS")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full PCA to analyze variance
    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_99 = np.argmax(cumulative_var >= 0.99) + 1

    print(f"  Total components      : {len(cumulative_var)}")
    print(f"  Components for 95% var: {n_95}")
    print(f"  Components for 99% var: {n_99}")
    print(f"  Top 5 explained variance: {np.round(pca_full.explained_variance_ratio_[:5], 4)}")

    # PCA charts
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("PCA Analysis", fontsize=16, fontweight="bold")

    # Chart 1: Explained variance per component
    n_show = min(20, len(pca_full.explained_variance_ratio_))
    axes[0].bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show],
                color="#1f77b4", edgecolor="white", alpha=0.8)
    axes[0].set_title("Explained variance per component")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")

    # Chart 2: Cumulative variance
    axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, "b-o", markersize=4)
    axes[1].axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
    axes[1].axhline(y=0.99, color="orange", linestyle="--", label="99% threshold")
    axes[1].axvline(x=n_95, color="red", linestyle=":", alpha=0.5)
    axes[1].axvline(x=n_99, color="orange", linestyle=":", alpha=0.5)
    axes[1].set_title("Cumulative explained variance")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].legend()

    # Chart 3: 2D PCA scatter
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X_scaled)
    scatter_colors = np.where(y == 1, "#d62728", "#2ca02c")
    axes[2].scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c="#2ca02c", alpha=0.3, s=10, label="Legit")
    axes[2].scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c="#d62728", alpha=0.8, s=30, label="Fraud", marker="x")
    axes[2].set_title("2D PCA projection")
    axes[2].set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})")
    axes[2].set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})")
    axes[2].legend()

    plt.tight_layout()
    pca_path = os.path.join(charts_dir, "03_pca_analysis.png")
    fig.savefig(pca_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ PCA charts saved → {pca_path}")

    # Return PCA-transformed data alongside originals for comparison
    pca_optimal = PCA(n_components=n_95)
    X_pca = pd.DataFrame(
        pca_optimal.fit_transform(X_scaled),
        columns=[f"PC{i+1}" for i in range(n_95)]
    )
    print(f"  PCA reduced: {X.shape[1]} features → {n_95} components (95% variance)")

    return X_pca, scaler, pca_optimal, n_95


# ═══════════════════════════════════════════════════════════════════
# 6. STRATIFIED SPLIT
# ═══════════════════════════════════════════════════════════════════
def stratified_split(X, y, X_pca):
    print("\n" + "=" * 70)
    print("  STEP 6: STRATIFIED TRAIN/TEST SPLIT (80/20)")
    print("=" * 70)

    # Use index-based splitting so PCA stays aligned
    indices = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=42
    )
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.15, stratify=y.iloc[idx_trainval], random_state=42
    )

    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y.iloc[idx_train], y.iloc[idx_val], y.iloc[idx_test]
    y_trainval = y.iloc[idx_trainval]  # for PCA model
    X_train_pca = X_pca.iloc[idx_trainval]  # PCA train+val (for comparison model)
    X_test_pca = X_pca.iloc[idx_test]

    print(f"  Train : {X_train.shape[0]:>5,} samples ({y_train.mean():.2%} fraud)")
    print(f"  Val   : {X_val.shape[0]:>5,} samples ({y_val.mean():.2%} fraud)")
    print(f"  Test  : {X_test.shape[0]:>5,} samples ({y_test.mean():.2%} fraud)")
    print(f"  ✓ Stratification preserved across all splits")

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_pca, X_test_pca, y_trainval


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
        use_label_encoder=False,
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
        n_jobs=-1,
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

    best_iter = model.best_iteration
    print(f"  ✓ Best iteration: {best_iter}")
    return model


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
# 10. PCA vs ORIGINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════
def train_and_compare_pca(X_train_pca, X_test_pca, y_train, y_test,
                           best_params, metrics_original, charts_dir):
    print("\n" + "=" * 70)
    print("  STEP 10: PCA MODEL — TRAIN & COMPARE")
    print("=" * 70)

    # Need a fresh train/val split for PCA data
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_train_pca, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    pca_model = xgb.XGBClassifier(
        **best_params,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",
    )
    pca_model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=0)

    metrics_pca, y_pred_pca, y_proba_pca = evaluate_model(
        pca_model, X_test_pca, y_test,
        list(X_test_pca.columns), charts_dir, label="PCA"
    )

    # Comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics_original.keys())
    x_pos = np.arange(len(metric_names))
    w = 0.35

    vals_orig = [metrics_original[m] for m in metric_names]
    vals_pca = [metrics_pca[m] for m in metric_names]

    bars1 = ax.bar(x_pos - w / 2, vals_orig, w, label="Original features", color="#1f77b4", edgecolor="white")
    bars2 = ax.bar(x_pos + w / 2, vals_pca, w, label="PCA features", color="#ff7f0e", edgecolor="white")

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.set_title("Original features vs PCA — metric comparison", fontsize=14, fontweight="bold")
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    comp_path = os.path.join(charts_dir, "05_pca_comparison.png")
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Comparison chart saved → {comp_path}")

    return metrics_pca, pca_model


# ═══════════════════════════════════════════════════════════════════
# 11. CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════════
def cross_validate_final(best_params, X, y, charts_dir):
    print("\n" + "=" * 70)
    print("  STEP 11: 5-FOLD CROSS VALIDATION")
    print("=" * 70)

    cv_model = xgb.XGBClassifier(
        **best_params,
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring_metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    cv_results = {}
    for metric in scoring_metrics:
        scores = cross_val_score(cv_model, X, y, cv=cv, scoring=metric, n_jobs=-1)
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
def save_artifacts(model, X, best_params, metrics_orig, metrics_pca,
                    cv_results, label_encoders, le_target, output_dir):
    print("\n" + "=" * 70)
    print("  STEP 12: SAVING MODEL & ARTIFACTS")
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
        "metrics_original_features": metrics_orig,
        "metrics_pca_features": metrics_pca,
        "cross_validation": {k: {"mean": v["mean"], "std": v["std"]}
                              for k, v in cv_results.items()},
        "label_encoders": {col: list(le.classes_) for col, le in label_encoders.items()},
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

    # 5. PCA
    X_pca, scaler, pca_model_obj, n_pca = run_pca_analysis(X, y, CHARTS_DIR)

    # 6. Stratified split
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_train_pca, X_test_pca, y_trainval) = stratified_split(X, y, X_pca)

    # 7. Hyperparameter tuning
    best_params, search = tune_hyperparameters(X_train, y_train)

    # 8. Train final model
    model = train_model(X_train, y_train, X_val, y_val, best_params)

    # 9. Evaluate (original features)
    metrics_orig, y_pred, y_proba = evaluate_model(
        model, X_test, y_test, list(X.columns), CHARTS_DIR, label="Original"
    )
    plot_evaluation_dashboard(y_test, y_pred, y_proba, metrics_orig,
                               list(X.columns), model, CHARTS_DIR)

    # 10. PCA comparison
    metrics_pca, pca_trained = train_and_compare_pca(
        X_train_pca, X_test_pca, y_trainval, y_test, best_params, metrics_orig, CHARTS_DIR
    )

    # 11. Cross-validation
    cv_results = cross_validate_final(best_params, X, y, CHARTS_DIR)

    # 12. Save
    save_artifacts(model, X, best_params, metrics_orig, metrics_pca,
                   cv_results, label_encoders, le_target, OUTPUT_DIR)

    # Final summary
    print("\n" + "█" * 70)
    print("  PIPELINE COMPLETE — SUMMARY")
    print("█" * 70)
    print(f"\n  {'Metric':<20} {'Original':>10} {'PCA':>10}")
    print(f"  {'─' * 42}")
    for k in metrics_orig:
        print(f"  {k:<20} {metrics_orig[k]:>10.4f} {metrics_pca[k]:>10.4f}")
    print(f"\n  Charts saved to: {CHARTS_DIR}")
    print(f"  Model saved to : {OUTPUT_DIR}")
    recommendation = "Original" if metrics_orig["f1_score"] >= metrics_pca["f1_score"] else "PCA"
    print(f"\n  ★ Recommended model: {recommendation} features (higher F1)")
    print("█" * 70)


if __name__ == "__main__":
    main()
