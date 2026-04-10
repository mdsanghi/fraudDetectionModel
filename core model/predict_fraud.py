"""
Fraud Detection — Real-Time Prediction Module
==============================================
Loads the frozen XGBoost model from xgb_fraud_model.json.
Label encoders are read directly from model_metadata.json (no joblib needed).

MINIMUM INPUT — caller only needs to provide:
    customer_id       : str   — must exist in Customers sheet
    beneficiary_id    : str   — must exist in Beneficiaries sheet
    amount_usd        : float — transfer amount
    transaction_type  : str   — "Wire Transfer" | "Bill Payment" | "Peer Transfer" | "Top-Up" | "Withdrawal"
    channel           : str   — "Mobile App" | "Web Browser" | "ATM" | "Branch"
    device_used       : str   — current device name

ALL other fraud signals are computed automatically:
    beneficiary_nationality  <- looked up from Beneficiaries sheet
    is_cross_nationality     <- customer.nationality != beneficiary.nationality
    customer_avg_amount      <- customer.typical_txn_amount
    amount_vs_avg_ratio      <- amount_usd / customer_avg_amount
    is_new_device            <- device_used != customer.primary_device
    is_new_beneficiary       <- beneficiary_id not in customer's past transactions
    is_high_risk_country     <- beneficiary nationality in {NG, RU, CN, BD}
    login_attempts_before    <- failed logins in last 24h from login history
    transaction_hour         <- from transaction_date (defaults to now)
    is_unusual_hour          <- transaction_hour in 0-4
"""

import os
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

RISK_THRESHOLDS         = {"low_max": 29, "medium_max": 69, "high_min": 70}
HIGH_RISK_NATIONALITIES = {"NG", "RU", "CN", "BD"}

DROP_COLS = [
    "transaction_id", "customer_id", "beneficiary_id",
    "full_name", "email", "phone", "account_number",
    "beneficiary_name", "ip_address",
    "beneficiary_nationality_name", "customer_nationality",
    "nationality_name", "nationality_name_cust",
    "fraud_type", "suspicious_flags", "status", "label",
]


class FraudPredictor:
    """
    Loads model + reference data once.
    score_transaction() accepts only 6 required inputs — everything else is derived.
    """

    def __init__(self, model_dir: str = None, data_path: str = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd        = os.getcwd()

        search_model = [d for d in [model_dir, script_dir, cwd,
                         os.path.join(cwd, "output"),
                         os.path.join(script_dir, "output")] if d]
        resolved_model_dir = next(
            (d for d in search_model
             if os.path.exists(os.path.join(d, "xgb_fraud_model.json"))), None)
        if not resolved_model_dir:
            raise FileNotFoundError(
                f"xgb_fraud_model.json not found. Searched: {search_model}")

        search_data = [d for d in [data_path, cwd, script_dir,
                        os.path.dirname(script_dir)] if d]
        resolved_data = next(
            (os.path.join(d, "fraud_detection_synthetic_data.xlsx")
             for d in search_data
             if os.path.exists(os.path.join(d, "fraud_detection_synthetic_data.xlsx"))),
            data_path if data_path and os.path.isfile(str(data_path)) else None)
        if not resolved_data:
            raise FileNotFoundError("fraud_detection_synthetic_data.xlsx not found. Pass data_path=")

        # Load model
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(resolved_model_dir, "xgb_fraud_model.json"))
        print(f"  OK Model loaded")

        # Load metadata + encoders from JSON (no joblib)
        with open(os.path.join(resolved_model_dir, "model_metadata.json")) as f:
            self.metadata = json.load(f)
        self.feature_order = self.metadata["features"]
        self.label_encoders = {
            col: {str(c): i for i, c in enumerate(classes)}
            for col, classes in self.metadata.get("label_encoders", {}).items()
        }
        print(f"  OK Label encoders: {len(self.label_encoders)} columns")

        # Load Excel reference data
        print(f"  Loading: {resolved_data}")
        xls = pd.ExcelFile(resolved_data)
        self.df_customers     = pd.read_excel(xls, "Customers")
        self.df_beneficiaries = pd.read_excel(xls, "Beneficiaries")
        self.df_logins        = pd.read_excel(xls, "Login_History")
        self.df_transactions  = pd.read_excel(xls, "Transactions")

        # Fast lookup dicts
        self._cust_map  = self.df_customers.set_index("customer_id").to_dict("index")
        self._bene_map  = self.df_beneficiaries.set_index("beneficiary_id").to_dict("index")
        self._cust_bene = (self.df_beneficiaries
                           .groupby("customer_id")["beneficiary_id"]
                           .apply(set).to_dict())

        # Account age
        self.df_customers["registration_date"] = pd.to_datetime(
            self.df_customers["registration_date"], errors="coerce")
        for cid, row in self._cust_map.items():
            reg = self.df_customers.loc[
                self.df_customers["customer_id"] == cid, "registration_date"]
            row["account_age_days"] = (
                int((pd.Timestamp.now() - reg.iloc[0]).days)
                if len(reg) > 0 and pd.notna(reg.iloc[0]) else 0)

        self._login_feats = self._build_login_features()
        self._bene_feats  = self._build_beneficiary_features()

        print(f"  OK {len(self.df_customers)} customers | "
              f"{len(self.df_logins)} logins | "
              f"{len(self.df_beneficiaries)} beneficiaries | "
              f"{len(self.df_transactions)} transactions")
        print(f"  OK Model expects {len(self.feature_order)} features\n")

    # ── Pre-compute aggregates ─────────────────────────────────────────────────

    def _build_login_features(self):
        df  = self.df_logins.copy()
        agg = df.groupby("customer_id").agg(
            total_logins        =("login_id",            "count"),
            failed_logins       =("login_status",        lambda x: (x != "Success").sum()),
            avg_session_mins    =("session_duration_mins","mean"),
            new_device_logins   =("is_new_device",       "sum"),
            new_location_logins =("is_new_location",     "sum"),
            mfa_usage_rate      =("mfa_used",            "mean"),
            unique_ips          =("ip_address",          "nunique"),
            unique_devices      =("device",              "nunique"),
            unique_countries    =("location_country",    "nunique"),
        ).reset_index()
        agg["failed_login_rate"]     = agg["failed_logins"] / agg["total_logins"]
        agg["has_repeated_failures"] = (agg["failed_logins"] >= 3).astype(int)
        last  = (df.sort_values("login_timestamp").groupby("customer_id")
                   .agg(last_login_country=("location_country","last")).reset_index())
        mode  = (df.groupby("customer_id")["location_country"]
                   .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
                   .reset_index().rename(columns={"location_country":"usual_login_country"}))
        agg   = agg.merge(last, on="customer_id", how="left")
        agg   = agg.merge(mode, on="customer_id", how="left")
        agg["login_location_shift"] = (
            agg["last_login_country"] != agg["usual_login_country"]).astype(int)
        agg.drop(columns=["last_login_country","usual_login_country"], inplace=True)
        agg["low_mfa_user"] = (agg["mfa_usage_rate"] < 0.3).astype(int)
        return agg.set_index("customer_id")

    def _build_beneficiary_features(self):
        df  = self.df_beneficiaries.copy()
        df["added_date"] = pd.to_datetime(df["added_date"], errors="coerce")
        agg = df.groupby("customer_id").agg(
            total_beneficiaries    =("beneficiary_id","count"),
            verified_beneficiaries =("is_verified",   "sum"),
            unique_ben_countries   =("country",       "nunique"),
            unique_ben_banks       =("bank_name",     "nunique"),
        ).reset_index()
        agg["unverified_ratio"]   = 1 - (agg["verified_beneficiaries"] / agg["total_beneficiaries"])
        agg["has_unverified_ben"] = (agg["unverified_ratio"] > 0).astype(int)
        now = pd.Timestamp.now()
        df["days_since_added"] = (now - df["added_date"]).dt.days
        recent = df.groupby("customer_id").agg(
            newest_ben_days   =("days_since_added","min"),
            ben_added_last_7d =("days_since_added",lambda x: (x <=  7).sum()),
            ben_added_last_30d=("days_since_added",lambda x: (x <= 30).sum()),
        ).reset_index()
        agg = agg.merge(recent, on="customer_id", how="left")
        agg["has_very_recent_ben"] = (agg["newest_ben_days"] <= 7).astype(int)
        return agg.set_index("customer_id")

    # ── Auto-derive all fraud flags ────────────────────────────────────────────

    def _derive_flags(self, txn: dict, cust: dict, bene: dict) -> dict:
        now = pd.Timestamp.now()
        d   = {}

        # Timestamp
        txn_dt = txn.get("transaction_date")
        if txn_dt is None:
            txn_dt = now
        elif isinstance(txn_dt, str):
            txn_dt = pd.to_datetime(txn_dt, errors="coerce")
            if pd.isna(txn_dt):
                txn_dt = now
        d["transaction_date"] = str(txn_dt)
        d["transaction_hour"] = int(txn_dt.hour) if hasattr(txn_dt, "hour") else now.hour
        d["is_unusual_hour"]  = int(d["transaction_hour"] in range(0, 5))

        # Amount
        avg = float(cust.get("typical_txn_amount", 0) or 1)
        amt = float(txn.get("amount_usd", 0))
        d["customer_avg_amount"] = avg
        d["amount_vs_avg_ratio"] = round(amt / avg, 4) if avg > 0 else 1.0

        # Device
        primary = str(cust.get("primary_device", ""))
        current = str(txn.get("device_used", ""))
        d["is_new_device"]   = int(bool(current) and current != primary)
        d["device_mismatch"] = d["is_new_device"]

        # Channel
        d["channel_mismatch"] = int(
            bool(txn.get("channel")) and
            txn.get("channel") != cust.get("preferred_channel", ""))

        # Beneficiary
        cid = txn.get("customer_id", "")
        bid = txn.get("beneficiary_id", "")
        d["is_new_beneficiary"] = int(bid not in self._cust_bene.get(cid, set()))

        bene_nat = str(bene.get("nationality", "")) if bene else ""
        d["beneficiary_nationality"] = bene_nat
        d["is_high_risk_country"]    = int(bene_nat in HIGH_RISK_NATIONALITIES)

        # Cross-nationality
        cust_nat = str(cust.get("nationality", ""))
        d["is_cross_nationality"] = int(
            bool(bene_nat) and bool(cust_nat) and bene_nat != cust_nat)

        # Login attempts last 24 h
        if cid and len(self.df_logins) > 0:
            logs = self.df_logins[self.df_logins["customer_id"] == cid].copy()
            logs["login_timestamp"] = pd.to_datetime(logs["login_timestamp"], errors="coerce")
            cutoff = now - timedelta(hours=24)
            d["login_attempts_before"] = int(
                len(logs[(logs["login_timestamp"] >= cutoff) &
                          (logs["login_status"] != "Success")]))
        else:
            d["login_attempts_before"] = 0

        # Compound
        d["cross_nat_new_ben"] = int(
            d["is_cross_nationality"] == 1 and d["is_new_beneficiary"] == 1)

        return d

    # ── Build full feature vector ──────────────────────────────────────────────

    def _build_features(self, txn: dict):
        cid  = txn.get("customer_id", "")
        bid  = txn.get("beneficiary_id", "")
        cust = self._cust_map.get(cid, {})
        bene = self._bene_map.get(bid, {})

        derived = self._derive_flags(txn, cust, bene)

        row = {}
        row.update(cust)
        row.update(txn)
        row.update(derived)        # derived wins — highest priority

        df = pd.DataFrame([row])

        # Merge login aggregates
        if cid in self._login_feats.index:
            for k, v in self._login_feats.loc[cid].to_dict().items():
                if k not in df.columns:
                    df[k] = v

        # Merge beneficiary aggregates
        if cid in self._bene_feats.index:
            for k, v in self._bene_feats.loc[cid].to_dict().items():
                if k not in df.columns:
                    df[k] = v

        # Spike signals
        amt = float(df["amount_usd"].iloc[0]) if "amount_usd" in df.columns else 0
        avg = float(df["customer_avg_amount"].iloc[0]) if "customer_avg_amount" in df.columns else 1
        avg = avg if avg > 0 else 1
        typ = float(df["typical_txn_amount"].iloc[0]) if "typical_txn_amount" in df.columns else 1
        typ = typ if typ > 0 else 1

        df["amount_ratio_to_avg"]     = round(amt / avg, 4)
        df["spike_2x"]                = int(amt / avg >= 2.0)
        df["spike_3x"]                = int(amt / avg >= 3.0)
        df["spike_5x"]                = int(amt / avg >= 5.0)
        df["amount_vs_typical_ratio"] = round(amt / typ, 4)
        df["spike_typical_2x"]        = int(amt / typ >= 2.0)
        df["spike_typical_5x"]        = int(amt / typ >= 5.0)

        # Hour / night
        hr = int(df["transaction_hour"].iloc[0]) if "transaction_hour" in df.columns else 12
        df["is_deep_night"] = int(hr in range(0, 5))

        # Login attempts
        attempts = int(df["login_attempts_before"].iloc[0]) if "login_attempts_before" in df.columns else 0
        df["high_login_attempts"] = int(attempts >= 3)

        # MFA
        df["high_value_txn"]  = int(amt > 400)
        low_mfa = int(df["low_mfa_user"].iloc[0]) if "low_mfa_user" in df.columns else 0
        df["no_mfa_high_value"] = int(low_mfa == 1 and amt > 400)

        # Customer risk
        df["is_high_risk_customer"] = int(
            str(df["risk_tier"].iloc[0]).lower() == "high" if "risk_tier" in df.columns else False)
        df["kyc_not_verified"] = int(
            str(df["kyc_status"].iloc[0]).lower() != "verified" if "kyc_status" in df.columns else False)
        df["account_not_active"] = int(
            str(df["account_status"].iloc[0]).lower() != "active" if "account_status" in df.columns else False)

        # Beneficiary risk combo
        is_new = int(df["is_new_beneficiary"].iloc[0]) if "is_new_beneficiary" in df.columns else 0
        has_unver = int(df["has_unverified_ben"].iloc[0]) if "has_unverified_ben" in df.columns else 0
        df["new_unverified_ben"] = int(is_new == 1 and has_unver == 1)

        # Signal count
        rule_flags = [
            "spike_2x","is_new_device","is_new_beneficiary","cross_nat_new_ben",
            "is_high_risk_country","is_deep_night","high_login_attempts",
            "no_mfa_high_value","is_high_risk_customer","kyc_not_verified",
            "new_unverified_ben","login_location_shift","device_mismatch",
        ]
        existing_flags = [f for f in rule_flags if f in df.columns]
        df["fraud_signal_count"] = int(df[existing_flags].fillna(0).sum(axis=1).iloc[0])
        df["multi_signal_alert"] = int(df["fraud_signal_count"].iloc[0] >= 3)

        # Drop ID/text cols
        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        # Datetime -> numeric
        for col in df.select_dtypes(include=["datetime64","datetimetz"]).columns:
            df[f"{col}_year"]      = df[col].dt.year
            df[f"{col}_month"]     = df[col].dt.month
            df[f"{col}_day"]       = df[col].dt.day
            df[f"{col}_hour"]      = df[col].dt.hour
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)

        # Encode categoricals
        if "fraud_signal_count" in df.columns and "fraud_signal_count" in self.label_encoders:
            enc = self.label_encoders["fraud_signal_count"]
            val = str(int(df["fraud_signal_count"].iloc[0]))
            df["fraud_signal_count"] = enc.get(val, -1)

        for col, enc_dict in self.label_encoders.items():
            if col == "fraud_signal_count":
                continue
            if col not in df.columns:
                continue
            val = str(df[col].iloc[0])
            df[col] = enc_dict.get(val, -1)

        # Align to model feature order
        for feat in self.feature_order:
            if feat not in df.columns:
                df[feat] = 0
        df = df[self.feature_order]

        # Final clean
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df, existing_flags

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_transaction(self, transaction: dict) -> dict:
        """
        Score a single transaction with minimal inputs.

        Required keys:
            customer_id, beneficiary_id, amount_usd,
            transaction_type, channel, device_used

        Optional:
            transaction_date  (defaults to now)

        All fraud flags are derived automatically.
        """
        features_df, flag_cols = self._build_features(transaction)

        probability = float(self.model.predict_proba(features_df)[0][1])
        ml_score    = round(probability * 100, 1)

        fired = [f for f in flag_cols
                 if f in features_df.columns and features_df[f].iloc[0] == 1]

        risk_level = (
            "Low"    if ml_score <= RISK_THRESHOLDS["low_max"]    else
            "Medium" if ml_score <= RISK_THRESHOLDS["medium_max"] else
            "High"
        )

        cid  = transaction.get("customer_id", "")
        bid  = transaction.get("beneficiary_id", "")
        derived = self._derive_flags(
            transaction,
            self._cust_map.get(cid, {}),
            self._bene_map.get(bid, {}),
        )

        return {
            "customer_id":    cid,
            "beneficiary_id": bid,
            "ml_probability": round(probability, 4),
            "ml_score":       ml_score,
            "prediction":     "Suspected" if probability >= 0.5 else "Not Suspected",
            "risk_level":     risk_level,
            "fraud_signals":  fired,
            "signal_count":   len(fired),
            "auto_derived": {
                "beneficiary_nationality": derived.get("beneficiary_nationality"),
                "is_cross_nationality":    derived.get("is_cross_nationality"),
                "is_new_device":           derived.get("is_new_device"),
                "is_new_beneficiary":      derived.get("is_new_beneficiary"),
                "is_high_risk_country":    derived.get("is_high_risk_country"),
                "login_attempts_before":   derived.get("login_attempts_before"),
                "transaction_hour":        derived.get("transaction_hour"),
                "is_unusual_hour":         derived.get("is_unusual_hour"),
                "customer_avg_amount":     derived.get("customer_avg_amount"),
                "amount_vs_avg_ratio":     derived.get("amount_vs_avg_ratio"),
                "device_mismatch":         derived.get("device_mismatch"),
                "channel_mismatch":        derived.get("channel_mismatch"),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  FRAUD PREDICTOR — MINIMAL INPUT DEMO")
    print("=" * 65)

    predictor = FraudPredictor()

    print("\n── Test 1: Suspicious (night, new device, large spike) ──")
    r = predictor.score_transaction({
        "customer_id":      "CUST1042",
        "beneficiary_id":   "BENE5001",
        "amount_usd":       4500.00,
        "transaction_type": "Wire Transfer",
        "channel":          "Web Browser",
        "device_used":      "Windows PC",
        "transaction_date": "2026-04-10 03:15:00",
    })
    print(f"  Prediction   : {r['prediction']}")
    print(f"  Risk Level   : {r['risk_level']}  (score: {r['ml_score']})")
    print(f"  Signals ({r['signal_count']})   : {r['fraud_signals']}")
    print("  Auto-derived flags:")
    for k, v in r["auto_derived"].items():
        print(f"    {k:<32} = {v}")

    print("\n── Test 2: Normal (small amount, daytime, known device) ──")
    r = predictor.score_transaction({
        "customer_id":      "CUST1010",
        "beneficiary_id":   "BENE5050",
        "amount_usd":       150.00,
        "transaction_type": "Bill Payment",
        "channel":          "Mobile App",
        "device_used":      "iPhone 14",
        "transaction_date": "2026-04-10 14:30:00",
    })
    print(f"  Prediction   : {r['prediction']}")
    print(f"  Risk Level   : {r['risk_level']}  (score: {r['ml_score']})")
    print(f"  Signals ({r['signal_count']})   : {r['fraud_signals']}")
    print("  Auto-derived flags:")
    for k, v in r["auto_derived"].items():
        print(f"    {k:<32} = {v}")

    print("\n── Test 3: Moderate spike, no transaction_date (defaults to now) ──")
    r = predictor.score_transaction({
        "customer_id":      "CUST1075",
        "beneficiary_id":   "BENE5200",
        "amount_usd":       600.00,
        "transaction_type": "Wire Transfer",
        "channel":          "Mobile App",
        "device_used":      "Samsung S23",
    })
    print(f"  Prediction   : {r['prediction']}")
    print(f"  Risk Level   : {r['risk_level']}  (score: {r['ml_score']})")
    print(f"  Signals ({r['signal_count']})   : {r['fraud_signals']}")
    print("  Auto-derived flags:")
    for k, v in r["auto_derived"].items():
        print(f"    {k:<32} = {v}")

    print("\n" + "=" * 65)
    print("  DEMO COMPLETE")
    print("=" * 65)