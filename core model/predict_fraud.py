"""
Fraud Detection — Real-Time Prediction Module
==============================================
Loads the frozen XGBoost model + Excel customer data.
Pass a transaction dict → get fraud probability + risk score.

Usage from app1.py:
    from predict import FraudPredictor
    predictor = FraudPredictor()
    result = predictor.score_transaction({
        "customer_id": "C0042",
        "beneficiary_id": "B00123",
        "amount_usd": 2500.00,
        "transaction_type": "transfer",
        "channel": "app",
        "device_used": "mobile",
        "transaction_hour": 3,
        ...
    })
    print(result)
    # {
    #   "ml_probability": 0.91,
    #   "ml_score": 91,
    #   "prediction": "suspected",
    #   "risk_level": "high",
    #   "fraud_signals": ["spike_5x", "is_deep_night", "device_mismatch"],
    #   "signal_count": 3
    # }
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════
# PATHS (adjust if your structure is different)
# ═══════════════════════════════════════════════════════════════════
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Try project root first, then CWD
for _base in [_PROJECT_ROOT, os.getcwd(), _SCRIPT_DIR]:
    if os.path.exists(os.path.join(_base, "fraud_detection_synthetic_data.xlsx")):
        BASE_DIR = _base
        break
else:
    BASE_DIR = _PROJECT_ROOT

DATA_PATH = os.path.join(BASE_DIR, "fraud_detection_synthetic_data.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DROP_COLS = [
    "transaction_id", "customer_id", "beneficiary_id",
    "full_name", "email", "phone", "account_number",
    "beneficiary_name", "ip_address",
    "beneficiary_nationality_name", "customer_nationality",
    "nationality_name", "nationality_name_cust",
    "fraud_type", "suspicious_flags", "status", "label",
]

# Risk level thresholds (matches fraud_rules.yaml)
RISK_THRESHOLDS = {"low_max": 29, "medium_max": 69, "high_min": 70}


class FraudPredictor:
    """
    Loads the frozen model + Excel reference data once.
    Call score_transaction() for each new transaction.
    """

    def __init__(self, model_dir=None, data_path=None):
        model_dir = model_dir or OUTPUT_DIR
        data_path = data_path or DATA_PATH

        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(model_dir, "xgb_fraud_model.json"))

        # Load label encoders
        self.label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))

        # Load metadata (feature order is critical)
        with open(os.path.join(model_dir, "model_metadata.json")) as f:
            self.metadata = json.load(f)
        self.feature_order = self.metadata["features"]

        # Load reference data from Excel (customer profiles, login history, beneficiaries)
        print("  Loading reference data from Excel...")
        self.df_customers = pd.read_excel(data_path, sheet_name="Customers")
        self.df_beneficiaries = pd.read_excel(data_path, sheet_name="Beneficiaries")
        self.df_logins = pd.read_excel(data_path, sheet_name="Login_History")

        # Pre-compute customer-level aggregates (done once at startup)
        self._login_features = self._build_login_features()
        self._ben_features = self._build_beneficiary_features()

        # Prepare customer lookup
        self.df_customers["registration_date"] = pd.to_datetime(
            self.df_customers["registration_date"], errors="coerce"
        )
        self.df_customers["account_age_days"] = (
            pd.Timestamp.now() - self.df_customers["registration_date"]
        ).dt.days

        print(f"  ✓ Model loaded ({len(self.feature_order)} features)")
        print(f"  ✓ Reference data: {len(self.df_customers)} customers, "
              f"{len(self.df_logins)} logins, {len(self.df_beneficiaries)} beneficiaries")

    # ── Pre-compute aggregates ───────────────────────────────────

    def _build_login_features(self):
        df = self.df_logins
        agg = df.groupby("customer_id").agg(
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
        agg["has_repeated_failures"] = (agg["failed_logins"] >= 3).astype(int)

        last_login = (
            df.sort_values("login_timestamp")
            .groupby("customer_id")
            .agg(last_login_country=("location_country", "last"))
            .reset_index()
        )
        mode_country = (
            df.groupby("customer_id")["location_country"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            .reset_index()
            .rename(columns={"location_country": "usual_login_country"})
        )
        agg = agg.merge(last_login, on="customer_id", how="left")
        agg = agg.merge(mode_country, on="customer_id", how="left")
        agg["login_location_shift"] = (agg["last_login_country"] != agg["usual_login_country"]).astype(int)
        agg.drop(columns=["last_login_country", "usual_login_country"], inplace=True)
        agg["low_mfa_user"] = (agg["mfa_usage_rate"] < 0.3).astype(int)
        return agg.set_index("customer_id")

    def _build_beneficiary_features(self):
        df = self.df_beneficiaries.copy()
        df["added_date"] = pd.to_datetime(df["added_date"], errors="coerce")

        agg = df.groupby("customer_id").agg(
            total_beneficiaries=("beneficiary_id", "count"),
            verified_beneficiaries=("is_verified", "sum"),
            unique_ben_countries=("country", "nunique"),
            unique_ben_banks=("bank_name", "nunique"),
        ).reset_index()
        agg["unverified_ratio"] = 1 - (agg["verified_beneficiaries"] / agg["total_beneficiaries"])
        agg["has_unverified_ben"] = (agg["unverified_ratio"] > 0).astype(int)

        now = pd.Timestamp.now()
        df["days_since_added"] = (now - df["added_date"]).dt.days
        recent_agg = df.groupby("customer_id").agg(
            newest_ben_days=("days_since_added", "min"),
            ben_added_last_7d=("days_since_added", lambda x: (x <= 7).sum()),
            ben_added_last_30d=("days_since_added", lambda x: (x <= 30).sum()),
        ).reset_index()
        agg = agg.merge(recent_agg, on="customer_id", how="left")
        agg["has_very_recent_ben"] = (agg["newest_ben_days"] <= 7).astype(int)
        return agg.set_index("customer_id")

    # ── Build feature vector for one transaction ─────────────────

    def _build_features(self, txn: dict) -> pd.DataFrame:
        """
        Takes a transaction dict, looks up customer/login/beneficiary data,
        engineers all features, returns a single-row DataFrame matching
        the model's expected feature order.
        """
        customer_id = txn["customer_id"]

        # Start with transaction fields
        row = dict(txn)

        # Merge customer profile
        cust = self.df_customers[self.df_customers["customer_id"] == customer_id]
        if len(cust) > 0:
            cust_dict = cust.iloc[0].to_dict()
            for k, v in cust_dict.items():
                if k != "customer_id" and k not in row:
                    row[k] = v
                elif k != "customer_id" and k in row and k + "_cust" not in row:
                    row[k + "_cust"] = v

        # Merge login features
        if customer_id in self._login_features.index:
            login_dict = self._login_features.loc[customer_id].to_dict()
            row.update(login_dict)

        # Merge beneficiary features
        if customer_id in self._ben_features.index:
            ben_dict = self._ben_features.loc[customer_id].to_dict()
            row.update(ben_dict)

        # Convert to single-row DataFrame
        df = pd.DataFrame([row])

        # ── Engineer fraud rule features ──
        if "amount_usd" in df.columns and "customer_avg_amount" in df.columns:
            avg = df["customer_avg_amount"].replace(0, np.nan)
            df["amount_ratio_to_avg"] = (df["amount_usd"] / avg).fillna(1.0)
            df["spike_2x"] = (df["amount_ratio_to_avg"] >= 2.0).astype(int)
            df["spike_3x"] = (df["amount_ratio_to_avg"] >= 3.0).astype(int)
            df["spike_5x"] = (df["amount_ratio_to_avg"] >= 5.0).astype(int)

        if "amount_usd" in df.columns and "typical_txn_amount" in df.columns:
            typ = df["typical_txn_amount"].replace(0, np.nan)
            df["amount_vs_typical_ratio"] = (df["amount_usd"] / typ).fillna(1.0)
            df["spike_typical_2x"] = (df["amount_vs_typical_ratio"] >= 2.0).astype(int)
            df["spike_typical_5x"] = (df["amount_vs_typical_ratio"] >= 5.0).astype(int)

        if "device_used" in df.columns and "primary_device" in df.columns:
            df["device_mismatch"] = (df["device_used"] != df["primary_device"]).astype(int)

        if "is_cross_nationality" in df.columns and "is_new_beneficiary" in df.columns:
            df["cross_nat_new_ben"] = ((df["is_cross_nationality"] == 1) & (df["is_new_beneficiary"] == 1)).astype(int)

        if "transaction_hour" in df.columns:
            df["is_deep_night"] = df["transaction_hour"].between(0, 5).astype(int)

        if "login_attempts_before" in df.columns:
            df["high_login_attempts"] = (df["login_attempts_before"] >= 3).astype(int)

        if "mfa_usage_rate" in df.columns and "amount_usd" in df.columns:
            median_amount = 200  # approximate from training data
            df["high_value_txn"] = (df["amount_usd"] > median_amount * 2).astype(int)
            if "low_mfa_user" in df.columns:
                df["no_mfa_high_value"] = ((df["low_mfa_user"] == 1) & (df["high_value_txn"] == 1)).astype(int)

        if "risk_tier" in df.columns:
            df["is_high_risk_customer"] = (df["risk_tier"] == "high").astype(int)
        if "kyc_status" in df.columns:
            df["kyc_not_verified"] = (df["kyc_status"] != "verified").astype(int)
        if "account_status" in df.columns:
            df["account_not_active"] = (df["account_status"] != "active").astype(int)

        if "is_new_beneficiary" in df.columns and "has_unverified_ben" in df.columns:
            df["new_unverified_ben"] = ((df["is_new_beneficiary"] == 1) & (df["has_unverified_ben"] == 1)).astype(int)

        if "channel" in df.columns and "preferred_channel" in df.columns:
            df["channel_mismatch"] = (df["channel"] != df["preferred_channel"]).astype(int)

        rule_flags = [
            "spike_2x", "is_new_device", "is_new_beneficiary", "cross_nat_new_ben",
            "is_high_risk_country", "is_deep_night", "high_login_attempts",
            "no_mfa_high_value", "is_high_risk_customer", "kyc_not_verified",
            "new_unverified_ben", "login_location_shift", "device_mismatch"
        ]
        existing_flags = [f for f in rule_flags if f in df.columns]
        df["fraud_signal_count"] = df[existing_flags].sum(axis=1)
        df["multi_signal_alert"] = (df["fraud_signal_count"] >= 3).astype(int)

        # ── Drop non-predictive columns ──
        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # ── Datetime → numeric ──
        for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)

        # ── Encode categoricals using saved encoders ──
        for col, le in self.label_encoders.items():
            if col in df.columns:
                val = str(df[col].iloc[0])
                if val in le.classes_:
                    df[col] = le.transform([val])[0]
                else:
                    df[col] = -1  # unseen category

        # ── Ensure all features exist, in correct order ──
        for feat in self.feature_order:
            if feat not in df.columns:
                df[feat] = 0

        df = df[self.feature_order]

        # ── Clean ──
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return df, existing_flags

    # ── Main scoring method ──────────────────────────────────────

    def score_transaction(self, transaction: dict) -> dict:
        """
        Score a single transaction.

        Args:
            transaction: dict with at minimum:
                - customer_id (str)
                - amount_usd (float)
                - transaction_type (str)
                - channel (str)
                - device_used (str)
                - transaction_hour (int)
                - is_new_device (int, 0/1)
                - is_new_beneficiary (int, 0/1)
                - is_cross_nationality (int, 0/1)
                - is_high_risk_country (int, 0/1)
                - is_unusual_hour (int, 0/1)
                - login_attempts_before (int)
                - beneficiary_nationality (str)
                - customer_avg_amount (float)
                - amount_vs_avg_ratio (float)

        Returns:
            dict with ml_probability, ml_score, prediction, risk_level,
            fraud_signals, signal_count
        """
        features_df, flag_cols = self._build_features(transaction)

        # Predict
        probability = float(self.model.predict_proba(features_df)[0][1])
        ml_score = round(probability * 100, 1)

        # Determine which fraud signals fired
        fired_signals = []
        for flag in flag_cols:
            if flag in features_df.columns and features_df[flag].iloc[0] == 1:
                fired_signals.append(flag)

        # Risk level
        if ml_score <= RISK_THRESHOLDS["low_max"]:
            risk_level = "low"
        elif ml_score <= RISK_THRESHOLDS["medium_max"]:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "customer_id": transaction["customer_id"],
            "ml_probability": round(probability, 4),
            "ml_score": ml_score,
            "prediction": "suspected" if probability >= 0.5 else "not_suspected",
            "risk_level": risk_level,
            "fraud_signals": fired_signals,
            "signal_count": len(fired_signals),
        }


# ═══════════════════════════════════════════════════════════════════
# DEMO — run this file directly to test
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  FRAUD PREDICTOR — DEMO")
    print("=" * 60)

    predictor = FraudPredictor()

    # Test 1: Suspicious transaction
    print("\n── Test 1: Highly suspicious transaction ──")
    result = predictor.score_transaction({
        "customer_id": "C0042",
        "beneficiary_id": "B00100",
        "beneficiary_nationality": "PK",
        "is_cross_nationality": 1,
        "transaction_type": "transfer",
        "amount_usd": 4500.00,
        "customer_avg_amount": 200.00,
        "amount_vs_avg_ratio": 22.5,
        "channel": "web",
        "device_used": "desktop",
        "is_new_device": 1,
        "is_new_beneficiary": 1,
        "is_high_risk_country": 1,
        "login_attempts_before": 5,
        "transaction_hour": 3,
        "is_unusual_hour": 1,
    })
    print(f"  Prediction   : {result['prediction']}")
    print(f"  ML Score     : {result['ml_score']}")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"  Signals ({result['signal_count']}): {result['fraud_signals']}")

    # Test 2: Normal transaction
    print("\n── Test 2: Normal transaction ──")
    result = predictor.score_transaction({
        "customer_id": "C0010",
        "beneficiary_id": "B00050",
        "beneficiary_nationality": "OM",
        "is_cross_nationality": 0,
        "transaction_type": "payment",
        "amount_usd": 150.00,
        "customer_avg_amount": 180.00,
        "amount_vs_avg_ratio": 0.83,
        "channel": "app",
        "device_used": "mobile",
        "is_new_device": 0,
        "is_new_beneficiary": 0,
        "is_high_risk_country": 0,
        "login_attempts_before": 1,
        "transaction_hour": 14,
        "is_unusual_hour": 0,
    })
    print(f"  Prediction   : {result['prediction']}")
    print(f"  ML Score     : {result['ml_score']}")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"  Signals ({result['signal_count']}): {result['fraud_signals']}")

    # Test 3: Borderline case
    print("\n── Test 3: Borderline — moderate spike, known device ──")
    result = predictor.score_transaction({
        "customer_id": "C0075",
        "beneficiary_id": "B00200",
        "beneficiary_nationality": "IN",
        "is_cross_nationality": 1,
        "transaction_type": "transfer",
        "amount_usd": 600.00,
        "customer_avg_amount": 250.00,
        "amount_vs_avg_ratio": 2.4,
        "channel": "app",
        "device_used": "mobile",
        "is_new_device": 0,
        "is_new_beneficiary": 1,
        "is_high_risk_country": 0,
        "login_attempts_before": 1,
        "transaction_hour": 10,
        "is_unusual_hour": 0,
    })
    print(f"  Prediction   : {result['prediction']}")
    print(f"  ML Score     : {result['ml_score']}")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"  Signals ({result['signal_count']}): {result['fraud_signals']}")

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)