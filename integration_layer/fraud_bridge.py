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
import threading
import traceback
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────

# Email address where Medium/High alerts are sent
COMPLIANCE_EMAIL = "msanghi.cdac@gmail.com"

# OMR to USD conversion rate (app1.py stores amount in INR, model expects USD)
# Flow: INR / exchange_rate = OMR  →  OMR * OMR_TO_USD = USD
OMR_TO_USD = 2.60

# Model / data file locations — auto-discovered from script dir + CWD.
# Override via environment variables if needed.
MODEL_DIR  = os.environ.get("FRAUD_MODEL_DIR",  os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.environ.get("FRAUD_DATA_PATH",  None)   # auto-found if None
RULES_PATH = os.environ.get("FRAUD_RULES_PATH", None)   # auto-found if None

# Maps app1.py beneficiary display names → IDs used in synthetic Excel data
# Update these to match actual IDs in your fraud_detection_synthetic_data.xlsx
BENEFICIARY_LOOKUP = {
    "Ashok UAE":    {"customer_id": "CUST1042", "beneficiary_id": "BENE5001"},
    "Canadarestca": {"customer_id": "CUST1042", "beneficiary_id": "BENE5002"},
    "Deepak India": {"customer_id": "CUST1042", "beneficiary_id": "BENE5003"},
    "Gaurav Khrgr": {"customer_id": "CUST1042", "beneficiary_id": "BENE5004"},
    "Harshit UAE":  {"customer_id": "CUST1042", "beneficiary_id": "BENE5005"},
}
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
        from predict_fraud import FraudPredictor
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

    Conversions:
        amount_inr / exchange_rate = amount_omr
        amount_omr * OMR_TO_USD    = amount_usd (model input)
        transfer_type              → "Wire Transfer" (mapped)
        channel                    → "Mobile App"   (app is mobile)
        device_used                → "iPhone 14"    (demo default)
    """
    bene_name    = td.get("beneficiary_name", "")
    ids          = BENEFICIARY_LOOKUP.get(
        bene_name,
        {"customer_id": DEFAULT_CUSTOMER_ID, "beneficiary_id": DEFAULT_BENEFICIARY_ID}
    )

    amount_inr    = float(td.get("amount_inr", 0))
    exchange_rate = float(td.get("exchange_rate", 239.58))
    amount_omr    = amount_inr / exchange_rate if exchange_rate > 0 else 0
    amount_usd    = round(amount_omr * OMR_TO_USD, 2)

    transfer_type = td.get("transfer_type", "Transfer Immediately")
    txn_type = "Wire Transfer" if "transfer" in transfer_type.lower() else "Bill Payment"

    return {
        "customer_id":      ids["customer_id"],
        "beneficiary_id":   ids["beneficiary_id"],
        "amount_usd":       amount_usd,
        "transaction_type": txn_type,
        "channel":          "Mobile App",
        "device_used":      "iPhone 14",
        "transaction_date": td.get("transfer_date", str(datetime.now().date())),
    }


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
    amount  = float(td.get("amount_inr", 0))
    country = td.get("beneficiary_country", "").lower()
    purpose = td.get("purpose", "").lower()

    if amount >= 20000:
        score += 30; triggered.append("High transfer amount (≥ INR 20,000)")
    elif amount >= 10000:
        score += 15; triggered.append("Moderate transfer amount (≥ INR 10,000)")
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


# ── Email builder ──────────────────────────────────────────────────────────────

def _build_alert_email(result: dict, td: dict, level: str) -> tuple:
    """Returns (subject, plain_text, html) for the compliance alert email."""
    cust_name    = td.get("customer_name", "Valued Customer")
    bene_name    = td.get("beneficiary_name", "Unknown")
    bene_bank    = td.get("beneficiary_bank", "")
    bene_country = td.get("beneficiary_country", "")
    amount_inr   = float(td.get("amount_inr", 0))
    exchange_rate= float(td.get("exchange_rate", 239.58))
    amount_omr   = amount_inr / exchange_rate if exchange_rate > 0 else 0
    acct_last4   = td.get("from_account", "")[-4:]
    txn_date     = td.get("transfer_date", str(datetime.now().date()))
    purpose      = td.get("purpose", "")
    auto         = result.get("auto_derived", {})

    colour   = "#A96900" if level == "Medium" else "#B42318"
    badge_bg = "#FFF8E7" if level == "Medium" else "#FDECEC"
    icon     = "⚠️" if level == "Medium" else "🚨"

    subject = (
        f"[NBO Alert] {icon} {level} Risk Fraud | "
        f"{cust_name} | OMR {amount_omr:.3f} → {bene_name}"
    )

    # Triggered signals as HTML rows
    signal_rows = "".join(
        f'<tr><td style="padding:5px 10px;font-size:13px;color:#444;">• {t}</td></tr>'
        for t in result.get("triggered", [])
    )

    # Auto-derived context as table rows
    ctx_items = [
        ("Beneficiary nationality", auto.get("beneficiary_nationality", "—")),
        ("Cross-nationality",       "Yes" if auto.get("is_cross_nationality") else "No"),
        ("New device used",         "Yes" if auto.get("is_new_device") else "No"),
        ("New beneficiary",         "Yes" if auto.get("is_new_beneficiary") else "No"),
        ("High-risk country",       "Yes" if auto.get("is_high_risk_country") else "No"),
        ("Transaction hour",        f"{auto.get('transaction_hour', '—')}:00"),
        ("Failed logins (24h)",     str(auto.get("login_attempts_before", 0))),
        ("Avg customer amount",     f"USD {auto.get('customer_avg_amount', 0):.2f}"),
        ("Amount vs avg ratio",     f"{auto.get('amount_vs_avg_ratio', 0):.2f}×"),
    ]
    ctx_rows = "".join(
        f'<tr style="background:{"#f9f9f9" if i % 2 == 0 else "#fff"};">'
        f'<td style="padding:7px 12px;font-size:13px;color:#777;width:42%;">{k}</td>'
        f'<td style="padding:7px 12px;font-size:13px;color:#111;font-weight:600;">{v}</td>'
        f'</tr>'
        for i, (k, v) in enumerate(ctx_items)
    )

    action_text = (
        "Review this transaction. Contact the customer to confirm if needed."
        if level == "Medium" else
        "IMMEDIATE ACTION REQUIRED. Consider placing a transaction hold and "
        "contacting the customer on their registered number. Escalate to the "
        "Fraud Investigation Unit if the customer is unreachable."
    )

    html = f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f4f4f4;margin:0;padding:20px;">
<div style="max-width:640px;margin:0 auto;background:#fff;border-radius:8px;
            overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,.12);">

  <div style="background:#1456D8;color:#fff;padding:20px 24px;">
    <div style="font-size:26px;font-weight:900;letter-spacing:1px;">NBO</div>
    <div style="font-size:11px;opacity:.75;margin-top:1px;">البنك الوطني العماني</div>
    <div style="font-size:15px;font-weight:600;margin-top:10px;">
      Fraud Detection Alert — Compliance Team
    </div>
  </div>

  <div style="background:{badge_bg};border-left:5px solid {colour};
              padding:12px 20px;font-size:14px;font-weight:700;color:{colour};">
    {icon}&nbsp; {level.upper()} RISK DETECTED
    &emsp;|&emsp; Composite: {result.get('composite_score', 0):.1f}/100
    &emsp;|&emsp; ML: {result.get('ml_score', 0):.1f}
    &emsp;|&emsp; YAML: {result.get('yaml_score_raw', 0)} pts
  </div>

  <div style="padding:24px;">

    <h3 style="color:#1456D8;margin:0 0 12px;font-size:15px;">Transaction Details</h3>
    <table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">
      <tr style="background:#EEF2FA;">
        <td style="padding:8px 12px;color:#555;font-size:13px;width:38%;">Customer</td>
        <td style="padding:8px 12px;color:#111;font-size:13px;font-weight:600;">
          {cust_name}&nbsp;(Account ···{acct_last4})</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;color:#555;font-size:13px;">Beneficiary</td>
        <td style="padding:8px 12px;color:#111;font-size:13px;font-weight:600;">
          {bene_name} — {bene_bank}, {bene_country}</td>
      </tr>
      <tr style="background:#EEF2FA;">
        <td style="padding:8px 12px;color:#555;font-size:13px;">Amount</td>
        <td style="padding:8px 12px;color:#1456D8;font-size:15px;font-weight:700;">
          INR {amount_inr:,.2f}&nbsp; ≈ &nbsp;OMR {amount_omr:.3f}</td>
      </tr>
      <tr>
        <td style="padding:8px 12px;color:#555;font-size:13px;">Date</td>
        <td style="padding:8px 12px;color:#111;font-size:13px;">{txn_date}</td>
      </tr>
      <tr style="background:#EEF2FA;">
        <td style="padding:8px 12px;color:#555;font-size:13px;">Purpose</td>
        <td style="padding:8px 12px;color:#111;font-size:13px;">{purpose}</td>
      </tr>
    </table>

    <h3 style="color:#1456D8;margin:20px 0 10px;font-size:15px;">Fraud Signals Triggered</h3>
    <table style="width:100%;border-collapse:collapse;background:#fafafa;border-radius:6px;">
      {signal_rows or '<tr><td style="padding:8px 12px;color:#888;font-size:13px;">No signals</td></tr>'}
    </table>

    <h3 style="color:#1456D8;margin:20px 0 10px;font-size:15px;">Auto-Derived Risk Context</h3>
    <table style="width:100%;border-collapse:collapse;border-radius:6px;overflow:hidden;">
      {ctx_rows}
    </table>

    <div style="background:{'#FFF3CD' if level == 'Medium' else '#FDECEC'};
                border-radius:8px;padding:14px 18px;margin-top:22px;
                border-left:4px solid {colour};">
      <div style="font-weight:700;color:{colour};font-size:14px;margin-bottom:6px;">
        {'⚠️ Recommended Action' if level == 'Medium' else '🚨 Immediate Action Required'}
      </div>
      <div style="font-size:13px;color:#444;line-height:1.6;">{action_text}</div>
    </div>

  </div>

  <div style="background:#f0f0f0;padding:14px 24px;font-size:11px;
              color:#888;text-align:center;">
    Auto-generated by NBO AI Fraud Detection System (POC v1.0)&emsp;|&emsp;
    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}&emsp;|&emsp;Do not reply.
  </div>

</div>
</body>
</html>"""

    plain = (
        f"NBO FRAUD ALERT — {level.upper()} RISK\n{'='*52}\n"
        f"Customer    : {cust_name} (Account ···{acct_last4})\n"
        f"Beneficiary : {bene_name} | {bene_bank} | {bene_country}\n"
        f"Amount      : INR {amount_inr:,.2f}  ≈  OMR {amount_omr:.3f}\n"
        f"Date        : {txn_date}\n"
        f"Purpose     : {purpose}\n\n"
        f"Composite Score : {result.get('composite_score', 0):.1f}\n"
        f"ML Score        : {result.get('ml_score', 0):.1f}\n"
        f"YAML Score      : {result.get('yaml_score_raw', 0)} pts\n\n"
        f"Signals:\n" +
        "\n".join(f"  • {t}" for t in result.get("triggered", [])) +
        f"\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    return subject, plain, html


def _send_async(result: dict, td: dict, level: str):
    """Send alert email in background thread."""
    try:
        from gmail_sender import send_email
        subject, plain, html = _build_alert_email(result, td, level)
        send_email(subject=subject, body_text=plain, body_html=html, to=COMPLIANCE_EMAIL)
        print(f"  [fraud_bridge] Email sent → {COMPLIANCE_EMAIL}  [{level}]")
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
          f"{transfer_data.get('beneficiary_name')} | "
          f"INR {transfer_data.get('amount_inr')} | "
          f"{transfer_data.get('purpose')}")

    # Load predictor
    predictor, err = _get_predictor()

    if predictor is None:
        print(f"  [fraud_bridge] Model unavailable ({err}) — using fallback")
        result = _fallback_risk(transfer_data)
    else:
        txn = _map_transfer_to_txn(transfer_data)
        print(f"  [fraud_bridge] Scoring: cust={txn['customer_id']}  "
              f"bene={txn['beneficiary_id']}  usd={txn['amount_usd']}")
        try:
            pred   = predictor.score_transaction(txn)
            result = _format_for_app(pred, transfer_data)
        except Exception as e:
            print(f"  [fraud_bridge] Prediction error: {e}")
            traceback.print_exc()
            result = _fallback_risk(transfer_data)

    level = result.get("level", "Low")
    print(f"  [fraud_bridge] Result: {level} | Score: {result.get('score')} | "
          f"Alarm: {result.get('alarm_reason', '-')}")

    # Dispatch email for Medium and High
    if level in ("Medium", "High"):
        t = threading.Thread(
            target=_send_async,
            args=(result, transfer_data, level),
            daemon=True,
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

    # Simulate what app1.py passes in
    sample_transfer = {
        "customer_name":      "MAYANK SANGHI",
        "from_account":       "10470376449016",
        "from_account_name":  "STAFF SAVINGS ACCOUNT OMR",
        "available_balance":  152.877,
        "beneficiary_name":   "Harshit UAE",
        "beneficiary_account":"AE920030012054234910001",
        "beneficiary_country":"UAE",
        "beneficiary_bank":   "ABU DHABI COMMERCIAL BANK",
        "beneficiary_branch": "DUBAI MAIN",
        "beneficiary_town":   "Dubai",
        "exchange_rate":      239.58,
        "amount_inr":         25000.0,
        "purpose":            "Gift",
        "remark":             "",
        "transfer_type":      "Transfer Immediately",
        "transfer_date":      "2026-04-10",
    }

    result = run_fraud_check(sample_transfer)

    print(f"\n  Level     : {result['level']}")
    print(f"  Score     : {result['score']}")
    print(f"  Email sent: {result['email_sent']}")
    print(f"  Signals   :")
    for t in result["triggered"]:
        print(f"    • {t}")