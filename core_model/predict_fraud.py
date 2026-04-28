"""
Fraud Detection — Real-Time Prediction Module
==============================================
Composite scoring:
    composite_score = ml_weight x ml_score  +  yaml_weight x yaml_score_normalised

Where:
    ml_score              = calibrated ML risk score  (0-100)
    yaml_score_normalised = (sum of fired rule scores / max possible score) * 100
    composite_score       = 0-100

Risk levels (applied to composite_score):
    Low    <  35
    Medium  35-69
    High   >= 70

Safety net — ALWAYS raises High alarm if:
    (a) composite_score >= 70, OR
    (b) yaml_score_raw  >= 70  (high YAML evidence even if ML is uncertain), OR
    (c) any CRITICAL rule fires (R03, R08, R16, R20)

MINIMUM CALLER INPUT:
    customer_id, beneficiary_id, amount_usd,
    transaction_type, channel, device_used
    (transaction_date optional — defaults to now)
"""

import os
import re
import json
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Thresholds ─────────────────────────────────────────────────────────────────
COMPOSITE_LOW    = 35
COMPOSITE_MEDIUM = 69
COMPOSITE_HIGH   = 70

# Raw YAML score above this always triggers High regardless of ML
YAML_HIGH_FLOOR  = 70
DATETIME_HINTS   = ("date", "timestamp", "dob", "birth")
MEDIUM_FLOOR_RULE_IDS = {"R05", "R18", "R21", "R23"}
EXCEL_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m-%d-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y",
    "%m-%d-%y %H:%M:%S",
    "%m-%d-%y %H:%M",
    "%m-%d-%y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m/%d/%y %H:%M:%S",
    "%m/%d/%y %H:%M",
    "%m/%d/%y",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
)


def _parse_excelish_dates(values) -> pd.Series:
    """Parse mixed Excel/text date formats, including Excel serial dates.

    ISO dates are parsed explicitly first so workbook values entered as
    yyyy-mm-dd are never interpreted through locale-dependent day/month rules.
    """
    series = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    text = series.astype("string").str.strip()
    blank_mask = text.isna() | text.eq("") | text.str.lower().isin({"nan", "nat", "none"})

    candidates = text.where(~blank_mask)
    normalized_text = (
        candidates.str.replace("T", " ", regex=False)
        .str.replace(r"\.\d+(?=($|[+-]\d{2}:?\d{2}$))", "", regex=True)
        .str.replace(r"(Z|[+-]\d{2}:?\d{2})$", "", regex=True)
        .str.strip()
    )

    for fmt in EXCEL_DATE_FORMATS:
        remaining = parsed.isna() & ~blank_mask
        if not remaining.any():
            break
        parsed_fmt = pd.to_datetime(normalized_text.where(remaining), format=fmt, errors="coerce")
        parsed.loc[parsed_fmt.notna()] = parsed_fmt[parsed_fmt.notna()]

    remaining = parsed.isna() & ~blank_mask
    if remaining.any():
        primary = pd.to_datetime(normalized_text.where(remaining), errors="coerce")
        parsed.loc[primary.notna()] = primary[primary.notna()]

    remaining = parsed.isna() & ~blank_mask
    if remaining.any():
        dayfirst = pd.to_datetime(normalized_text.where(remaining), errors="coerce", dayfirst=True)
        parsed.loc[dayfirst.notna()] = dayfirst[dayfirst.notna()]

    numeric = pd.to_numeric(text, errors="coerce")
    serial_mask = parsed.isna() & numeric.notna() & numeric.between(1, 60000)
    if serial_mask.any():
        serial_dates = pd.to_datetime(
            numeric[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
        )
        parsed.loc[serial_dates.index] = serial_dates

    return parsed


def _parse_excelish_date(value):
    return _parse_excelish_dates(pd.Series([value])).iloc[0]

HIGH_RISK_NATIONALITIES = {"NG", "RU", "CN", "BD"}

# Rules tagged critical — any single fire forces High alarm
CRITICAL_RULE_IDS = {"R03", "R08", "R16", "R20", "R22"}

DROP_COLS = [
    "transaction_id", "customer_id", "beneficiary_id",
    "full_name", "email", "phone", "account_number",
    "beneficiary_name", "ip_address",
    "beneficiary_nationality_name", "customer_nationality",
    "nationality_name", "nationality_name_cust",
    "fraud_type", "suspicious_flags", "status", "label",
]


# ══════════════════════════════════════════════════════════════════════════════
# YAML Rule Engine
# ══════════════════════════════════════════════════════════════════════════════

def _load_rules(yaml_path: str) -> list:
    """Load rules from fraud_rules.yaml. Falls back to built-in rules if yaml unavailable."""
    if not os.path.exists(yaml_path):
        return []
    if _YAML_AVAILABLE:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return data.get("rules", [])
    # Fallback: parse YAML manually for simple key:value structure
    rules = []
    current = {}
    with open(yaml_path) as f:
        in_rules = False
        for line in f:
            stripped = line.strip()
            if stripped == "rules:":
                in_rules = True
                continue
            if not in_rules:
                continue
            if stripped.startswith("- id:"):
                if current:
                    rules.append(current)
                current = {"id": stripped.split(":", 1)[1].strip()}
            elif ":" in stripped and not stripped.startswith("#"):
                k, v = stripped.split(":", 1)
                k, v = k.strip(), v.strip()
                if k in ("score",):
                    try:
                        current[k] = int(v)
                    except ValueError:
                        current[k] = v
                else:
                    current[k] = v
    if current:
        rules.append(current)
    return rules


def _eval_rule(when_clause: str, ctx: dict) -> bool:
    """
    Evaluate a YAML 'when' clause against a context dict.
    Supports: ==, !=, >=, <=, >, <, 'and', 'or', true/false, quoted strings.
    """
    # Normalise booleans and string literals
    expr = when_clause.strip()
    expr = re.sub(r'\btrue\b',  '1', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bfalse\b', '0', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\band\b',   ' and ', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bor\b',    ' or ',  expr, flags=re.IGNORECASE)

    # Build a safe eval namespace from the context
    ns = {}
    for k, v in ctx.items():
        if isinstance(v, (int, float, bool)):
            ns[k] = v
        elif isinstance(v, str):
            ns[k] = v
        else:
            try:
                ns[k] = float(v)
            except (TypeError, ValueError):
                ns[k] = str(v) if v is not None else ""

    try:
        return bool(eval(expr, {"__builtins__": {}}, ns))  # noqa: S307
    except Exception:
        return False


def evaluate_yaml_rules(rules: list, ctx: dict) -> tuple:
    """
    Evaluate all rules against context dict.
    Returns:
        fired_rules     : list of dicts  {id, name, score, severity}
        yaml_score_raw  : int  (sum of fired rule scores)
        max_score       : int  (sum of all rule scores — used for normalisation)
        critical_fired  : bool (any CRITICAL rule triggered)
    """
    fired       = []
    total_score = 0
    max_score   = sum(r.get("score", 0) for r in rules)
    critical    = False

    for rule in rules:
        rid   = rule.get("id", "")
        when  = rule.get("when", "")
        score = int(rule.get("score", 0))
        sev   = rule.get("severity", "medium")
        name  = rule.get("name", rid)

        if not when:
            continue

        try:
            triggered = _eval_rule(when, ctx)
        except Exception:
            triggered = False

        if triggered:
            fired.append({"id": rid, "name": name, "score": score, "severity": sev})
            total_score += score
            if rid in CRITICAL_RULE_IDS:
                critical = True

    return fired, total_score, max_score, critical


# ══════════════════════════════════════════════════════════════════════════════
# FraudPredictor
# ══════════════════════════════════════════════════════════════════════════════

class FraudPredictor:
    """
    Composite fraud scorer:
        composite_score = 0.60 * ml_score + 0.40 * yaml_normalised
    Minimum caller input: customer_id, beneficiary_id, amount_usd,
                          transaction_type, channel, device_used
    """

    def __init__(self, model_dir: str = None, data_path: str = None,
                 rules_path: str = None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd        = os.getcwd()
        print("\n[predict_fraud] Initializing FraudPredictor")
        print(f"[predict_fraud] cwd={cwd}")

        # ── Resolve model dir ──────────────────────────────────────────────────
        search_model = [d for d in [model_dir, script_dir, cwd,
                         os.path.join(cwd, "output"),
                         os.path.join(script_dir, "output")] if d]
        resolved_model_dir = next(
            (d for d in search_model
             if os.path.exists(os.path.join(d, "xgb_fraud_model.json"))), None)
        if not resolved_model_dir:
            raise FileNotFoundError(
                f"xgb_fraud_model.json not found. Searched: {search_model}")
        print(f"[predict_fraud] resolved_model_dir={resolved_model_dir}")

        # ── Resolve data path ──────────────────────────────────────────────────
        search_data = [d for d in [data_path, cwd, script_dir,
                        os.path.dirname(script_dir)] if d]
        resolved_data = next(
            (os.path.join(d, "fraud_detection_synthetic_data.xlsx")
             for d in search_data
             if os.path.exists(
                 os.path.join(d, "fraud_detection_synthetic_data.xlsx"))),
            data_path if data_path and os.path.isfile(str(data_path)) else None)
        if not resolved_data:
            raise FileNotFoundError(
                "fraud_detection_synthetic_data.xlsx not found. Pass data_path=")
        print(f"[predict_fraud] resolved_data={resolved_data}")

        # ── Resolve YAML rules path ────────────────────────────────────────────
        search_rules = [rules_path] if rules_path else []
        search_rules += [script_dir, cwd,
                         os.path.join(cwd, "output"),
                         os.path.join(script_dir, "output"),
                         os.path.join(cwd, "rule engine")]  # Add rule engine directory
        resolved_rules = next(
            (os.path.join(d, "fraud_rules.yaml")
             for d in search_rules
             if d and os.path.exists(os.path.join(d, "fraud_rules.yaml"))),
            None)
        print(f"[predict_fraud] resolved_rules={resolved_rules}")

        # ── Load XGBoost model ─────────────────────────────────────────────────
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(resolved_model_dir, "xgb_fraud_model.json"))
        print(f"  OK  XGBoost model loaded")

        # ── Load metadata + label encoders ────────────────────────────────────
        with open(os.path.join(resolved_model_dir, "model_metadata.json")) as f:
            self.metadata = json.load(f)
        self.feature_order = self.metadata["features"]
        self.label_encoders = {
            col: {str(c): i for i, c in enumerate(classes)}
            for col, classes in self.metadata.get("label_encoders", {}).items()
        }
        self.inference_config = self.metadata.get("inference_config", {})
        self.probability_calibration = self.metadata.get("probability_calibration", {})
        scoring_config = self.inference_config.get("scoring_config", {})
        self.composite_weights = scoring_config.get("weights", {"ml": 0.7, "yaml": 0.3})
        self.thresholds = scoring_config.get(
            "thresholds", {"medium_min": COMPOSITE_LOW, "high_min": COMPOSITE_HIGH}
        )
        print(f"  OK  Label encoders: {len(self.label_encoders)} columns")

        # ── Load YAML rules ────────────────────────────────────────────────────
        if resolved_rules:
            self.rules = _load_rules(resolved_rules)
            self.max_yaml_score = sum(r.get("score", 0) for r in self.rules)
            print(f"  OK  YAML rules: {len(self.rules)} rules loaded, "
                  f"max score = {self.max_yaml_score}")
        else:
            self.rules = []
            self.max_yaml_score = 1
            print("  WARN  fraud_rules.yaml not found — YAML scoring disabled")

        # ── Load reference Excel sheets ────────────────────────────────────────
        print(f"  Loading: {resolved_data}")
        xls = pd.ExcelFile(resolved_data)
        self.df_customers     = pd.read_excel(xls, "Customers")
        self.df_beneficiaries = pd.read_excel(xls, "Beneficiaries")
        self.df_logins        = pd.read_excel(xls, "Login_History")
        self.df_transactions  = pd.read_excel(xls, "Transactions")

        for col in ("date_of_birth", "registration_date"):
            if col in self.df_customers.columns:
                self.df_customers[col] = _parse_excelish_dates(self.df_customers[col])
        if "added_date" in self.df_beneficiaries.columns:
            self.df_beneficiaries["added_date"] = _parse_excelish_dates(
                self.df_beneficiaries["added_date"]
            )
        if "login_timestamp" in self.df_logins.columns:
            self.df_logins["login_timestamp"] = _parse_excelish_dates(
                self.df_logins["login_timestamp"]
            )
        if "transaction_date" in self.df_transactions.columns:
            self.df_transactions["transaction_date"] = _parse_excelish_dates(
                self.df_transactions["transaction_date"]
            )

        self._cust_map  = self.df_customers.set_index("customer_id").to_dict("index")
        self._bene_map  = self.df_beneficiaries.set_index("beneficiary_id").to_dict("index")
        self._cust_bene = (self.df_beneficiaries
                           .groupby("customer_id")["beneficiary_id"]
                           .apply(set).to_dict())
        self._customer_avg_amounts = (
            self.df_transactions.groupby("customer_id")["amount_usd"].mean().to_dict()
        )

        for cid, row in self._cust_map.items():
            reg = self.df_customers.loc[
                self.df_customers["customer_id"] == cid, "registration_date"]
            row["account_age_days"] = (
                int((pd.Timestamp.now() - reg.iloc[0]).days)
                if len(reg) > 0 and pd.notna(reg.iloc[0]) else 0)

        self._login_feats = self._build_login_features()
        self._bene_feats  = self._build_beneficiary_features()
        self.high_value_amount_threshold = float(
            self.inference_config.get(
                "high_value_amount_threshold",
                round(float(self.df_transactions["amount_usd"].median()) * 2, 2),
            )
        )

        print(f"  OK  {len(self.df_customers)} customers | "
              f"{len(self.df_logins)} logins | "
              f"{len(self.df_beneficiaries)} beneficiaries")
        print(f"  OK  High-value threshold: {self.high_value_amount_threshold:.2f} USD\n")

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
        last = (df.sort_values("login_timestamp").groupby("customer_id")
                  .agg(last_login_country=("location_country","last")).reset_index())
        mode = (df.groupby("customer_id")["location_country"]
                  .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
                  .reset_index()
                  .rename(columns={"location_country":"usual_login_country"}))
        agg = agg.merge(last, on="customer_id", how="left")
        agg = agg.merge(mode, on="customer_id", how="left")
        agg["login_location_shift"] = (
            agg["last_login_country"] != agg["usual_login_country"]).astype(int)
        agg.drop(columns=["last_login_country","usual_login_country"], inplace=True)
        agg["low_mfa_user"] = (agg["mfa_usage_rate"] < 0.3).astype(int)
        return agg.set_index("customer_id")

    def _build_beneficiary_features(self):
        df  = self.df_beneficiaries.copy()
        df["added_date"] = _parse_excelish_dates(df["added_date"])
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

    def _coerce_datetime_like_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert parseable date-like strings into datetime columns."""
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            if not pd.api.types.is_object_dtype(df[col]):
                continue
            if not any(hint in col.lower() for hint in DATETIME_HINTS):
                continue
            parsed = _parse_excelish_dates(df[col])
            if parsed.notna().mean() >= 0.6:
                df[col] = parsed
        return df

    # ── Auto-derive all fraud flags ────────────────────────────────────────────

    def _derive_flags(self, txn: dict, cust: dict, bene: dict) -> dict:
        now = pd.Timestamp.now()
        d   = {}

        # Timestamp
        txn_dt = txn.get("transaction_date")
        if txn_dt is None:
            txn_dt = now
        else:
            txn_dt = _parse_excelish_date(txn_dt)
            if pd.isna(txn_dt):
                txn_dt = now
        d["transaction_date"] = str(txn_dt)
        d["transaction_hour"] = int(txn_dt.hour) if hasattr(txn_dt, "hour") else now.hour
        d["is_unusual_hour"]  = int(d["transaction_hour"] in range(0, 5))

        # Amount
        avg = float(self._customer_avg_amounts.get(txn.get("customer_id", ""), 0) or
                    cust.get("typical_txn_amount", 0) or 1)
        amt = float(txn.get("amount_usd", 0))
        d["customer_avg_amount"]  = avg
        d["amount_vs_avg_ratio"]  = round(amt / avg, 4) if avg > 0 else 1.0

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
            logs["login_timestamp"] = _parse_excelish_dates(logs["login_timestamp"])
            cutoff = txn_dt - timedelta(hours=24)
            d["login_attempts_before"] = int(len(
                logs[(logs["login_timestamp"] >= cutoff) &
                     (logs["login_timestamp"] <= txn_dt) &
                     (logs["login_status"] != "Success")]))
        else:
            d["login_attempts_before"] = 0

        # Recent login location shift
        if cid and len(self.df_logins) > 0:
            logs = self.df_logins[self.df_logins["customer_id"] == cid].copy()
            logs["login_timestamp"] = _parse_excelish_dates(logs["login_timestamp"])
            logs = logs[logs["login_timestamp"] <= txn_dt]
            last_log = logs.sort_values("login_timestamp").iloc[-1] if len(logs) > 0 else None
            d["recent_login_new_location"] = int(
                last_log is not None and
                bool(last_log.get("is_new_location", False))
            )
            # Recent MFA
            d["recent_mfa_used"] = int(
                last_log is not None and
                bool(last_log.get("mfa_used", True))
            )
        else:
            d["recent_login_new_location"] = 0
            d["recent_mfa_used"]           = 1

        # Compound
        d["cross_nat_new_ben"] = int(
            d["is_cross_nationality"] == 1 and d["is_new_beneficiary"] == 1)

        # Beneficiary age (days since added)
        if bene:
            added = bene.get("added_date", None)
            if added:
                try:
                    added_dt = _parse_excelish_date(added)
                    d["beneficiary_age_days"] = int((txn_dt - added_dt).days) if not pd.isna(added_dt) else 999
                except Exception:
                    d["beneficiary_age_days"] = 999
            else:
                d["beneficiary_age_days"] = 999
        else:
            d["beneficiary_age_days"] = 999

        # Beneficiary verified flag for YAML rules
        d["beneficiary_is_verified"] = int(bool(bene.get("is_verified", True))) if bene else 1

        # Customer fields for YAML rules (mapped to YAML field names)
        d["customer_risk_tier"]     = str(cust.get("risk_tier", "Low"))
        d["customer_kyc_status"]    = str(cust.get("kyc_status", "Verified"))
        d["customer_account_status"]= str(cust.get("account_status", "Active"))
        reg_dt = _parse_excelish_date(cust.get("registration_date", None)) if cust else pd.NaT
        if pd.notna(reg_dt):
            d["account_age_days"] = int((txn_dt - reg_dt).days)
        else:
            d["account_age_days"] = int(cust.get("account_age_days", 999)) if cust.get("account_age_days", None) is not None else 999
        d["account_recent_5d"] = int(d["account_age_days"] <= 5)
        d["beneficiary_recent_5d"] = int(d["beneficiary_age_days"] <= 5)
        d["recent_account_and_beneficiary_5d"] = int(
            d["account_recent_5d"] == 1 and d["beneficiary_recent_5d"] == 1
        )

        return d

    # ── Build XGBoost feature vector ───────────────────────────────────────────

    def _build_features(self, txn: dict, derived: dict):
        cid  = txn.get("customer_id", "")
        cust = self._cust_map.get(cid, {})
        row  = {}
        row.update(cust)
        row.update(txn)
        row.update(derived)    # derived wins

        df = pd.DataFrame([row])

        # Login aggregates
        if cid in self._login_feats.index:
            for k, v in self._login_feats.loc[cid].to_dict().items():
                if k not in df.columns:
                    df[k] = v

        # Beneficiary aggregates
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

        hr = int(df["transaction_hour"].iloc[0]) if "transaction_hour" in df.columns else 12
        df["is_deep_night"]       = int(hr in range(0, 5))
        attempts = int(df["login_attempts_before"].iloc[0]) if "login_attempts_before" in df.columns else 0
        df["high_login_attempts"] = int(attempts >= 3)
        df["high_value_txn"]      = int(amt > self.high_value_amount_threshold)
        low_mfa = int(df["low_mfa_user"].iloc[0]) if "low_mfa_user" in df.columns else 0
        df["no_mfa_high_value"]   = int(low_mfa == 1 and amt > self.high_value_amount_threshold)

        df["is_high_risk_customer"] = int(
            str(df["risk_tier"].iloc[0]).lower() == "high" if "risk_tier" in df.columns else False)
        df["kyc_not_verified"]  = int(
            str(df["kyc_status"].iloc[0]).lower() != "verified" if "kyc_status" in df.columns else False)
        df["account_not_active"] = int(
            str(df["account_status"].iloc[0]).lower() != "active" if "account_status" in df.columns else False)
        df["account_recent_5d"] = int(
            float(df["account_age_days"].iloc[0]) <= 5 if "account_age_days" in df.columns else 0
        )
        df["beneficiary_recent_5d"] = int(
            float(df["beneficiary_age_days"].iloc[0]) <= 5 if "beneficiary_age_days" in df.columns else 0
        )
        df["recent_account_and_beneficiary_5d"] = int(
            df["account_recent_5d"].iloc[0] == 1 and df["beneficiary_recent_5d"].iloc[0] == 1
        )

        is_new   = int(df["is_new_beneficiary"].iloc[0]) if "is_new_beneficiary" in df.columns else 0
        has_unv  = int(df["has_unverified_ben"].iloc[0]) if "has_unverified_ben" in df.columns else 0
        df["new_unverified_ben"] = int(is_new == 1 and has_unv == 1)

        rule_flags = [
            "spike_2x","is_new_device","is_new_beneficiary","cross_nat_new_ben",
            "is_high_risk_country","is_deep_night","high_login_attempts",
            "no_mfa_high_value","is_high_risk_customer","kyc_not_verified",
            "new_unverified_ben","login_location_shift","device_mismatch",
            "account_recent_5d","beneficiary_recent_5d","recent_account_and_beneficiary_5d",
        ]
        existing_flags = [f for f in rule_flags if f in df.columns]
        df["fraud_signal_count"] = int(df[existing_flags].fillna(0).sum(axis=1).iloc[0])
        df["multi_signal_alert"] = int(df["fraud_signal_count"].iloc[0] >= 3)

        for col in DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        df = self._coerce_datetime_like_columns(df)

        for col in df.select_dtypes(include=["object", "category"]).columns:
            numeric_candidate = pd.to_numeric(df[col], errors="coerce")
            if numeric_candidate.notna().mean() >= 0.98:
                df[col] = numeric_candidate

        for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df.drop(columns=[col], inplace=True)

        if "fraud_signal_count" in df.columns and "fraud_signal_count" in self.label_encoders:
            enc = self.label_encoders["fraud_signal_count"]
            val = str(int(df["fraud_signal_count"].iloc[0]))
            df["fraud_signal_count"] = enc.get(val, 0)

        unknown_encodings = {}
        for col, enc_dict in self.label_encoders.items():
            if col not in df.columns:
                continue
            raw_val = str(df[col].iloc[0])
            encoded = enc_dict.get(raw_val)
            if encoded is None:
                unknown_encodings[col] = raw_val
                encoded = 0
            df[col] = encoded

        for feat in self.feature_order:
            if feat not in df.columns:
                df[feat] = 0
        df = df[self.feature_order]
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if unknown_encodings:
            print(f"[predict_fraud] fallback encodings applied: {unknown_encodings}")

        return df, existing_flags

    # ── Composite scoring ──────────────────────────────────────────────────────

    def _calibrate_probability(self, raw_probability: float) -> float:
        """Apply saved isotonic calibration if available."""
        meta = self.probability_calibration
        if meta.get("method") != "isotonic":
            return raw_probability
        x_thresholds = np.array(meta.get("x_thresholds", []), dtype=float)
        y_thresholds = np.array(meta.get("y_thresholds", []), dtype=float)
        if len(x_thresholds) == 0 or len(y_thresholds) == 0:
            return raw_probability
        isotonic_probability = float(np.interp(raw_probability, x_thresholds, y_thresholds))
        blend_alpha = float(meta.get("blend_alpha", 0.65))
        return float(blend_alpha * raw_probability + (1 - blend_alpha) * isotonic_probability)

    def _ml_risk_score(self, calibrated_probability: float) -> float:
        """Convert calibrated probability into the ML component of the risk score."""
        return round(calibrated_probability * 100, 2)

    def _composite_score(self, ml_score: float,
                         yaml_score_raw: int, max_yaml: int) -> float:
        """
        composite = ml_weight * ml_score + yaml_weight * yaml_normalised
        ml_score        : 0-100
        yaml_score_raw  : sum of fired rule scores
        max_yaml        : sum of all rule scores (normalisation base)
        """
        yaml_norm = (yaml_score_raw / max_yaml * 100) if max_yaml > 0 else 0
        ml_weight = float(self.composite_weights.get("ml", 0.7))
        yaml_weight = float(self.composite_weights.get("yaml", 0.3))
        return round(ml_weight * ml_score + yaml_weight * yaml_norm, 2)
        # Take only ML score for testing
        # return round(ml_score, 2)

    def _risk_level(self, composite: float, yaml_raw: int,
                    critical_fired: bool, yaml_fired: list | None = None,
                    auto_derived: dict | None = None) -> tuple:
        """
        Determine final risk level.
        Safety net ensures High is ALWAYS raised when:
          (a) composite >= 70
          (b) yaml_raw  >= YAML_HIGH_FLOOR  (strong rule evidence alone)
          (c) any CRITICAL rule fired
        Returns (risk_level, reason)
        """
        # Safety net checks first
        if critical_fired:
            return "High", "critical_rule_fired"
        if yaml_raw >= YAML_HIGH_FLOOR:
            return "High", "yaml_high_floor_exceeded"
        # Normal composite thresholds
        high_min = float(self.thresholds.get("high_min", COMPOSITE_HIGH))
        medium_min = float(self.thresholds.get("medium_min", COMPOSITE_LOW))
        if composite >= high_min:
            return "High", "composite_score"
        if composite >= medium_min:
            return "Medium", "composite_score"
        fired_ids = {rule.get("id", "") for rule in (yaml_fired or [])}
        auto = auto_derived or {}
        medium_floor = bool(
            fired_ids.intersection(MEDIUM_FLOOR_RULE_IDS)
            or auto.get("is_new_beneficiary")
            or auto.get("account_recent_5d")
            or auto.get("beneficiary_recent_5d")
        )
        if medium_floor:
            return "Medium", "medium_floor_signal"
        return "Low", "composite_score"

    # ── Public API ─────────────────────────────────────────────────────────────

    def score_transaction(self, transaction: dict) -> dict:
        """
        Score a single transaction with minimal inputs.

        Required: customer_id, beneficiary_id, amount_usd,
                  transaction_type, channel, device_used
        Optional: transaction_date  (defaults to now)

        Returns full scoring breakdown including composite score,
        ML score, YAML score, fired rules, and auto-derived flags.
        """
        cid  = transaction.get("customer_id", "")
        bid  = transaction.get("beneficiary_id", "")
        cust = self._cust_map.get(cid, {})
        bene = self._bene_map.get(bid, {})
        print(f"\n[predict_fraud] score_transaction called for customer_id={cid}, beneficiary_id={bid}")
        print(f"[predict_fraud] incoming transaction={transaction}")

        # ── Step 1: Derive all flags automatically ─────────────────────────────
        derived = self._derive_flags(transaction, cust, bene)
        print(f"[predict_fraud] derived flags={derived}")

        # ── Step 2: XGBoost ML score ───────────────────────────────────────────
        features_df, ml_flag_cols = self._build_features(transaction, derived)
        print(f"[predict_fraud] feature shape={features_df.shape}")
        raw_ml_probability = float(self.model.predict_proba(features_df)[0][1])
        ml_probability = self._calibrate_probability(raw_ml_probability)
        ml_score = self._ml_risk_score(ml_probability)

        ml_fired_flags = [
            f for f in ml_flag_cols
            if f in features_df.columns and features_df[f].iloc[0] == 1
        ]

        # ── Step 3: YAML rule engine score ────────────────────────────────────
        # Build context for rule evaluation (all derived + transaction fields)
        rule_ctx = {}
        rule_ctx.update(cust)
        rule_ctx.update(transaction)
        rule_ctx.update(derived)

        yaml_fired, yaml_raw, max_yaml, critical_fired = evaluate_yaml_rules(
            self.rules, rule_ctx)
        max_yaml = max_yaml if max_yaml > 0 else self.max_yaml_score

        yaml_norm = round((yaml_raw / max_yaml * 100), 2) if max_yaml > 0 else 0
        print(
            "[predict_fraud] scoring summary: "
            f"raw_ml_probability={raw_ml_probability:.4f}, "
            f"ml_probability={ml_probability:.4f}, ml_score={ml_score}, "
            f"yaml_raw={yaml_raw}, yaml_norm={yaml_norm}, "
            f"critical_fired={critical_fired}, ml_flags={ml_fired_flags}"
        )

        # ── Step 4: Composite score (60% ML + 40% YAML) ───────────────────────
        composite = self._composite_score(ml_score, yaml_raw, max_yaml)

        # ── Step 5: Final risk level with safety net ───────────────────────────
        risk_level, alarm_reason = self._risk_level(
            composite, yaml_raw, critical_fired, yaml_fired=yaml_fired, auto_derived=derived)
        print(
            "[predict_fraud] final decision: "
            f"composite={composite}, risk_level={risk_level}, alarm_reason={alarm_reason}"
        )

        return {
            # Core output
            "customer_id":    cid,
            "beneficiary_id": bid,
            "prediction":     "Suspected" if ml_probability >= 0.5 else "Not Suspected",
            "risk_level":     risk_level,
            "alarm_reason":   alarm_reason,

            # Score breakdown
            "composite_score":     composite,
            "ml_score":            ml_score,
            "ml_probability":      round(ml_probability, 4),
            "raw_ml_probability":  round(raw_ml_probability, 4),
            "yaml_score_raw":      yaml_raw,
            "yaml_score_normalised": yaml_norm,
            "yaml_max_possible":   max_yaml,

            # Weights applied
            "weights": self.composite_weights,

            # Signals
            "ml_signals":     ml_fired_flags,
            "yaml_rules_fired": [
                f"{r['id']} {r['name']} (+{r['score']} pts, {r['severity']})"
                for r in yaml_fired
            ],
            "signal_count":   len(ml_fired_flags),
            "yaml_rule_count": len(yaml_fired),

            # Transparency — what was auto-derived
            "auto_derived": {
                "beneficiary_nationality":  derived.get("beneficiary_nationality"),
                "is_cross_nationality":     derived.get("is_cross_nationality"),
                "is_new_device":            derived.get("is_new_device"),
                "is_new_beneficiary":       derived.get("is_new_beneficiary"),
                "is_high_risk_country":     derived.get("is_high_risk_country"),
                "login_attempts_before":    derived.get("login_attempts_before"),
                "transaction_hour":         derived.get("transaction_hour"),
                "is_unusual_hour":          derived.get("is_unusual_hour"),
                "customer_avg_amount":      derived.get("customer_avg_amount"),
                "amount_vs_avg_ratio":      derived.get("amount_vs_avg_ratio"),
                "beneficiary_age_days":     derived.get("beneficiary_age_days"),
                "recent_login_new_location":derived.get("recent_login_new_location"),
                "recent_mfa_used":          derived.get("recent_mfa_used"),
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  FRAUD PREDICTOR — COMPOSITE SCORING DEMO")
    print("  Formula: composite = 0.60 x ML_score + 0.40 x YAML_normalised")
    print("=" * 70)

    predictor = FraudPredictor()

    def print_result(label, r):
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        print(f"  Prediction        : {r['prediction']}")
        print(f"  Risk Level        : {r['risk_level']}  [{r['alarm_reason']}]")
        print(f"  ─ Composite Score : {r['composite_score']:.1f} / 100")
        print(f"  ─ ML Score        : {r['ml_score']:.1f}  (weight 60%)")
        print(f"  ─ YAML Score      : {r['yaml_score_raw']} raw "
              f"→ {r['yaml_score_normalised']:.1f} normalised  (weight 40%)")
        print(f"  ML Signals  ({r['signal_count']})  : {r['ml_signals']}")
        print(f"  YAML Rules  ({r['yaml_rule_count']})  :")
        for rr in r["yaml_rules_fired"]:
            print(f"      {rr}")
        print(f"  Auto-derived flags:")
        for k, v in r["auto_derived"].items():
            print(f"      {k:<36} = {v}")

    print_result("Test 1: Low-risk customer — regular beneficiary, daytime, normal amount",
        predictor.score_transaction({
            "customer_id":      "CUST1157",
            "beneficiary_id":   "BENE5771",
            "amount_usd":       1617.45,
            "transaction_type": "Bill Payment",
            "channel":          "Mobile App",
            "device_used":      "iPhone 14",
            "transaction_date": "2026-04-10 13:30:00",
        })
    )

    print_result("Test 2: Medium-risk customer — regular beneficiary, daytime, normal amount",
        predictor.score_transaction({
            "customer_id":      "CUST1157",
            "beneficiary_id":   "BENE5771",
            "amount_usd":       4617.45,
            "transaction_type": "Peer Transfer",
            "channel":          "Mobile App",
            "device_used":      "iPhone 14",
            "transaction_date": "2026-04-10 3:30:00",
        })
    )

    print_result("Test 3: HIGH-risk customer — regular beneficiary, daytime, normal amount",
        predictor.score_transaction({
            "customer_id":      "CUST1157",
            "beneficiary_id":   "BENE5772",
            "amount_usd":       4617.45,
            "transaction_type": "Peer Transfer",
            "channel":          "Mobile App",
            "device_used":      "iPhone 7",
            "transaction_date": "2026-04-10 3:30:00",
        })
    )

    

    print(f"\n{'='*70}")
    print("  DEMO COMPLETE")
    print(f"{'='*70}")
