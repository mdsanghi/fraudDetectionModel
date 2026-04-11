"""
fraud_bridge.py
===============
Bridge between app1.py (Streamlit UI) and predict_fraud.py (ML + YAML engine).

Called from app1.py after OTP submission, replacing the local compute_risk() call.

HOW TO WIRE INTO app1.py
─────────────────────────
1. Add at the top of app1.py:
       from fraud_bridge import run_fraud_check

2. Replace the OTP submit handler:
       OLD:  st.session_state.risk_result = compute_risk(st.session_state.transfer_data)
       NEW:  st.session_state.risk_result = run_fraud_check(st.session_state.transfer_data)

3. The result screen needs NO changes — it still reads result["level"], result["score"],
   result["triggered"] exactly as before.

ACTION MATRIX:
    Low    → no action
    Medium → send fraud alert email to compliance team (async, non-blocking)
    High   → send escalated fraud alert email to compliance team (async, non-blocking)
"""

import os
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CORE_MODEL_DIR = PROJECT_ROOT / "core model"
if str(CORE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_MODEL_DIR))

from email_config.gmail_sender import send_email, send_fraud_alert
from core_model.predict_fraud import FraudPredictor

# ── Configuration ──────────────────────────────────────────────────────────────

# Email address where Medium/High alerts are sent
COMPLIANCE_EMAIL = "msanghi.cdac@gmail.com"

# OMR to USD conversion rate kept only for display conversions in alert emails.
OMR_TO_USD = 2.60

# Model / data file locations — auto-discovered from script dir + CWD.
# Override via environment variables if needed.
MODEL_DIR  = os.environ.get("FRAUD_MODEL_DIR",  os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.environ.get("FRAUD_DATA_PATH",  None)   # auto-found if None
RULES_PATH = os.environ.get("FRAUD_RULES_PATH", None)   # auto-found if None

DEFAULT_CUSTOMER_ID    = "CUST1042"
DEFAULT_BENEFICIARY_ID = "BENE5001"

# Human-readable labels for ML signal codes
ML_SIGNAL_LABELS = {
    "spike_2x":              "Amount is 2x above customer average",
    "spike_3x":              "Amount is 3x above customer average",
    "spike_5x":              "Amount is 5x above customer average — extreme spike",
    "is_new_device":         "Transaction from an unrecognised device",
    "is_new_beneficiary":    "First-ever transfer to this beneficiary",
    "cross_nat_new_ben":     "New beneficiary of different nationality",
    "is_high_risk_country":  "Beneficiary is in a high-risk country",
    "is_deep_night":         "Transaction at an unusual late-night hour",
    "high_login_attempts":   "Multiple failed login attempts before transaction",
    "no_mfa_high_value":     "High-value transfer without MFA verification",
    "is_high_risk_customer": "Customer is flagged as High risk tier",
    "kyc_not_verified":      "Customer KYC is not fully verified",
    "new_unverified_ben":    "Unverified new beneficiary",
    "login_location_shift":  "Login from a new country before transaction",
    "device_mismatch":       "Device differs from customer primary device",
    "channel_mismatch":      "Channel differs from customer preferred channel",
}

# ── Singleton predictor (loaded once at first call) ───────────────────────────
_predictor       = None
_predictor_error = None


def _get_predictor():
    global _predictor, _predictor_error
    if _predictor is not None:
        return _predictor, None
    if _predictor_error is not None:
        return None, _predictor_error
    try:
        _predictor = FraudPredictor(
            model_dir  = MODEL_DIR,
            data_path  = DATA_PATH,
            rules_path = RULES_PATH,
        )
        return _predictor, None
    except Exception as e:
        _predictor_error = str(e)
        return None, _predictor_error


# ── Map app1.py transfer_data → predict_fraud input ───────────────────────────

def _map_transfer_to_txn(td: dict) -> dict:
    """
    Converts app1.py transfer_data fields to the 6-field minimal input
    expected by FraudPredictor.score_transaction().

    Assumptions:
        amount_usd is already provided by app1.py
        transfer_type              → "Wire Transfer" (mapped)
        channel and device_used are taken from the input when provided
    """
    customer_id = str(td.get("customer_id", DEFAULT_CUSTOMER_ID)).strip() or DEFAULT_CUSTOMER_ID
    beneficiary_id = str(td.get("beneficiary_id", DEFAULT_BENEFICIARY_ID)).strip() or DEFAULT_BENEFICIARY_ID

    amount_usd = round(float(td.get("amount_usd", 0)), 2)
    channel = str(td.get("channel", "Mobile App"))
    device_used = str(td.get("device_used", "iPhone 14"))

    transfer_type = td.get("transfer_type", "Transfer Immediately")
    txn_type = "Wire Transfer" if "transfer" in transfer_type.lower() else "Bill Payment"

    return {
        "customer_id":      customer_id,
        "beneficiary_id":   beneficiary_id,
        "amount_usd":       amount_usd,
        "transaction_type": txn_type,
        "channel":          channel,
        "device_used":      device_used,
        "transaction_date": td.get("transaction_timestamp", datetime.now().isoformat()),
    }


def _enrich_transfer_data(transfer_data: dict, predictor: FraudPredictor | None = None) -> dict:
    """Populate customer and beneficiary details from Excel-backed predictor data."""
    td = dict(transfer_data)
    customer_id = str(td.get("customer_id", "")).strip()
    beneficiary_id = str(td.get("beneficiary_id", "")).strip()

    if predictor is None:
        return td

    if customer_id:
        customer = predictor._cust_map.get(customer_id, {})
        if customer:
            td.setdefault("customer_name", str(customer.get("full_name", "")))
            td.setdefault("customer_email", str(customer.get("email", "")))
            td.setdefault("email", str(customer.get("email", "")))
            td.setdefault("country_of_residence", str(customer.get("country_of_residence", "")))
            td.setdefault("risk_tier", str(customer.get("risk_tier", "")))
            td.setdefault("kyc_status", str(customer.get("kyc_status", "")))
            td.setdefault("account_status", str(customer.get("account_status", "")))
            td.setdefault(
                "from_account_name",
                f"{str(customer.get('preferred_channel', 'Mobile App')).upper()} ACCOUNT",
            )

            if not td.get("from_account"):
                digits = "".join(ch for ch in customer_id if ch.isdigit())
                td["from_account"] = (digits + "10470376449016")[:14]

    if not beneficiary_id:
        return td

    beneficiary = predictor._bene_map.get(beneficiary_id, {})
    if not beneficiary:
        return td

    td.setdefault("beneficiary_name", str(beneficiary.get("beneficiary_name", "")))
    td.setdefault("beneficiary_account", str(beneficiary.get("account_number", "")))
    td.setdefault("beneficiary_country", str(beneficiary.get("country", "")))
    td.setdefault("beneficiary_bank", str(beneficiary.get("bank_name", "")))
    td.setdefault("beneficiary_branch", str(beneficiary.get("country", "")))
    td.setdefault("beneficiary_town", str(beneficiary.get("country", "")))

    if "exchange_rate" not in td or td.get("exchange_rate") in (None, "", 0, 0.0):
        td["exchange_rate"] = (
            239.58 if str(beneficiary.get("country", "")).strip().lower() == "india" else 1.0
        )

    return td


# ── Format predict_fraud output → app1.py result screen format ────────────────

def _format_for_app(pred: dict, td: dict) -> dict:
    """
    Translates FraudPredictor.score_transaction() output into the format
    app1.py result screen expects:
        result["level"]     : "Low" | "Medium" | "High"
        result["score"]     : int (0-100)
        result["triggered"] : list of human-readable strings
    """
    level = pred.get("risk_level", "Low")
    score = int(round(pred.get("composite_score", 0)))
    auto  = pred.get("auto_derived", {})

    triggered = []

    # ML signals → readable labels
    for sig in pred.get("ml_signals", []):
        label = ML_SIGNAL_LABELS.get(sig, sig.replace("_", " ").title())
        triggered.append(label)

    # YAML rules — extract just rule id + name
    for rule_str in pred.get("yaml_rules_fired", []):
        # rule_str format: "R04 new_device_used (+22 pts, high)"
        parts = rule_str.split(" ", 2)
        if len(parts) >= 2:
            triggered.append(f"Rule {parts[0]}: {parts[1].replace('_', ' ').title()}")

    # Auto-derived context additions
    if auto.get("is_cross_nationality"):
        triggered.append("Cross-nationality transfer detected")
    if auto.get("is_unusual_hour"):
        triggered.append(f"Unusual transaction hour ({auto.get('transaction_hour', '?')}:00)")
    attempts = auto.get("login_attempts_before", 0)
    if attempts and int(attempts) > 0:
        triggered.append(f"{attempts} failed login attempt(s) in last 24h")

    # Deduplicate, preserve order
    seen, unique = set(), []
    for t in triggered:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if not unique:
        unique = ["No significant fraud signals detected"]

    return {
        # ── app1.py result screen keys (must match exactly) ──
        "level":     level,
        "score":     score,
        "triggered": unique,

        # ── Extended fields (available for richer display) ──
        "composite_score":       pred.get("composite_score"),
        "ml_score":              pred.get("ml_score"),
        "ml_probability":        pred.get("ml_probability"),
        "yaml_score_raw":        pred.get("yaml_score_raw"),
        "yaml_score_normalised": pred.get("yaml_score_normalised"),
        "ml_signals":            pred.get("ml_signals", []),
        "yaml_rules_fired":      pred.get("yaml_rules_fired", []),
        "auto_derived":          auto,
        "alarm_reason":          pred.get("alarm_reason", ""),
        "prediction":            pred.get("prediction", ""),
        "email_sent":            False,
    }


# ── Simple fallback (when model unavailable) ───────────────────────────────────

def _fallback_risk(td: dict) -> dict:
    """Original compute_risk() logic — used if ML model fails to load."""
    score, triggered = 0, []
    amount  = float(td.get("amount_usd", 0))
    country = td.get("beneficiary_country", "").lower()
    purpose = td.get("purpose", "").lower()

    if amount >= 5000:
        score += 30; triggered.append("High transfer amount (≥ USD 5,000)")
    elif amount >= 2000:
        score += 15; triggered.append("Moderate transfer amount (≥ USD 2,000)")
    if country not in ["oman", "india"]:
        score += 35; triggered.append("Less common beneficiary country")
    elif country == "india":
        score += 10; triggered.append("Cross-border transfer to India")
    if purpose in ["gift", "personal", "others"]:
        score += 10; triggered.append("Generic transfer purpose")
    if not str(td.get("remark", "")).strip():
        score += 5; triggered.append("No remark provided")

    level = "High" if score >= 45 else "Medium" if score >= 20 else "Low"
    return {
        "level": level, "score": score,
        "triggered": triggered or ["No significant signals"],
        "composite_score": float(score), "ml_score": 0.0,
        "ml_probability": 0.0, "yaml_score_raw": score,
        "yaml_score_normalised": 0.0, "ml_signals": [],
        "yaml_rules_fired": [], "auto_derived": {},
        "alarm_reason": "fallback_simple_rules",
        "prediction": "Suspected" if score >= 45 else "Not Suspected",
        "email_sent": False,
    }


def _send_async(result: dict, td: dict, level: str):
    """Send alert email in background thread."""
    try:
        recipient_email = (
            td.get("customer_email")
            or td.get("email")
            or COMPLIANCE_EMAIL
        )

        if level == "High":
            amount_usd = float(td.get("amount_usd", 0))

            send_fraud_alert(
                recipient_email=recipient_email,
                customer_name=td.get("customer_name", "Valued Customer"),
                account_last_4=str(td.get("from_account", ""))[-4:],
                amount=amount_usd,
                merchant_name=td.get("beneficiary_name", "Unknown Beneficiary"),
                transaction_date=td.get("transfer_date", str(datetime.now().date())),
                currency="USD",
                bank_name="National bank of oman - HIGH RISK ALERT",
            )
        elif level == "Medium":
            send_fraud_alert(
                recipient_email=recipient_email,
                customer_name=td.get("customer_name", "Valued Customer"),
                account_last_4=str(td.get("from_account", ""))[-4:],
                amount=float(td.get("amount_usd", 0)),
                merchant_name=td.get("beneficiary_name", "Unknown Beneficiary"),
                transaction_date=td.get("transfer_date", str(datetime.now().date())),
                currency="USD",
                bank_name="National bank of oman  - MEDIUM RISK ALERT",
            )

        print(f"  [fraud_bridge] Email sent → {recipient_email}  [{level}]")
    except Exception as e:
        print(f"  [fraud_bridge] Email failed: {e}")
        traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def run_fraud_check(transfer_data: dict) -> dict:
    """
    Main entry point called from app1.py after OTP submission.

    Args:
        transfer_data: st.session_state.transfer_data from app1.py

    Returns:
        dict with keys: level, score, triggered  (+ extended scoring fields)

    Side effects:
        Medium risk → sends fraud alert email (async)
        High risk   → sends escalated fraud alert email (async)
        Low risk    → no email
    """
    print(f"\n[fraud_bridge] Checking: "
          f"{transfer_data.get('beneficiary_id') or transfer_data.get('beneficiary_name')} | "
          f"USD {transfer_data.get('amount_usd')} | "
          f"{transfer_data.get('purpose')}")

    # Load predictor
    predictor, err = _get_predictor()

    if predictor is None:
        print(f"  [fraud_bridge] Model unavailable ({err}) — using fallback")
        enriched_transfer_data = _enrich_transfer_data(transfer_data, None)
        result = _fallback_risk(enriched_transfer_data)
    else:
        enriched_transfer_data = _enrich_transfer_data(transfer_data, predictor)
        txn = _map_transfer_to_txn(enriched_transfer_data)
        print(f"  [fraud_bridge] Scoring: cust={txn['customer_id']}  "
              f"bene={txn['beneficiary_id']}  usd={txn['amount_usd']}")
        try:
            pred   = predictor.score_transaction(txn)
            result = _format_for_app(pred, enriched_transfer_data)
        except Exception as e:
            print(f"  [fraud_bridge] Prediction error: {e}")
            traceback.print_exc()
            result = _fallback_risk(enriched_transfer_data)

    level = result.get("level", "Low")
    print(f"  [fraud_bridge] Result: {level} | Score: {result.get('score')} | "
          f"Alarm: {result.get('alarm_reason', '-')}")

    # Dispatch email for Medium and High
    if level in ("Medium", "High"):
        t = threading.Thread(
            target=_send_async,
            args=(result, enriched_transfer_data, level),
            daemon=False,
        )
        t.start()
        result["email_sent"] = True
        print(f"  [fraud_bridge] Alert email queued → {COMPLIANCE_EMAIL}")
    else:
        result["email_sent"] = False
        print(f"  [fraud_bridge] Low risk — no email")

    return result


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  fraud_bridge.py — self test")
    print("=" * 60)

    sample_transfer = {
        "customer_id":         "CUST1000",
        "beneficiary_id":      "BENE5001",
        "amount_usd":          363.82,
        "channel":             "Mobile App",
        "device_used":         "MacBook Pro",
        "purpose":             "Family maintenance",
        "remark":              "",
        "transfer_type":       "Transfer Immediately",
        "transfer_date":       str(datetime.now().date()),
    }

    result = run_fraud_check(sample_transfer)

    print(f"\n  Level     : {result['level']}")
    print(f"  Score     : {result['score']}")
    print(f"  Email sent: {result['email_sent']}")
    print(f"  Signals   :")
    for t in result["triggered"]:
        print(f"    • {t}")
