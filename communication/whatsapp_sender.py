"""
whatsapp_sender.py
------------------
Send WhatsApp messages using Twilio's WhatsApp API without extra dependencies.

Configuration:
    WHATSAPP_ACCOUNT_SID
    WHATSAPP_AUTH_TOKEN
    WHATSAPP_FROM_NUMBER
    WHATSAPP_DEFAULT_COUNTRY_CODE
    WHATSAPP_TO_NUMBER

Example:
    from communication.whatsapp_sender import send_low_risk_action_required
"""

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Optional


# WhatsApp configuration - Update these with your Twilio WhatsApp credentials
WHATSAPP_ACCOUNT_SID = "AC0841282cfe626714aacba1c52e0301eb"         # Replace with your Twilio Account SID
WHATSAPP_AUTH_TOKEN = "0bc460e6b80518a19f0e6837946e03d1"          # Replace with your Twilio Auth Token
WHATSAPP_FROM_NUMBER = "+96899858946"         # Replace with your Twilio WhatsApp-enabled number
WHATSAPP_DEFAULT_COUNTRY_CODE = "+968"   # Used when phone numbers do not include a country code
WHATSAPP_TO_NUMBER = "+918446387628"           # Optional: force all messages to a single test number


def _normalize_phone(number: str, default_country_code: Optional[str] = None) -> str:
    raw = str(number or "").strip()
    if not raw:
        raise ValueError("Missing WhatsApp destination phone number.")

    if raw.lower().startswith("whatsapp:"):
        raw = raw.split(":", 1)[1]

    digits = re.sub(r"[^\d+]", "", raw)
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    elif not digits.startswith("+"):
        cc = (default_country_code or WHATSAPP_DEFAULT_COUNTRY_CODE or "").strip()
        if cc and not cc.startswith("+"):
            cc = "+" + cc
        digits = f"{cc}{digits}" if cc else f"+{digits}"

    normalized_digits = "+" + re.sub(r"\D", "", digits)
    if len(re.sub(r"\D", "", normalized_digits)) < 8:
        raise ValueError(f"WhatsApp phone number looks invalid: {number}")
    return f"whatsapp:{normalized_digits}"


def send_whatsapp_message(
    to_number: str,
    message: str,
    *,
    account_sid: Optional[str] = None,
    auth_token: Optional[str] = None,
    from_number: Optional[str] = None,
) -> dict:
    """Send a WhatsApp message via Twilio REST API and return the response body."""
    sid = (account_sid or WHATSAPP_ACCOUNT_SID).strip()
    token = (auth_token or WHATSAPP_AUTH_TOKEN).strip()
    sender = (from_number or WHATSAPP_FROM_NUMBER).strip()
    recipient = (WHATSAPP_TO_NUMBER or to_number).strip()

    if not sid:
        raise ValueError("Missing WHATSAPP_ACCOUNT_SID.")
    if not token:
        raise ValueError("Missing WHATSAPP_AUTH_TOKEN.")
    if not sender:
        raise ValueError("Missing WHATSAPP_FROM_NUMBER.")
    if not message.strip():
        raise ValueError("WhatsApp message body cannot be empty.")

    sender_uri = _normalize_phone(sender)
    recipient_uri = _normalize_phone(recipient)

    payload = urllib.parse.urlencode(
        {
            "From": sender_uri,
            "To": recipient_uri,
            "Body": message,
        }
    ).encode("utf-8")

    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    request = urllib.request.Request(url, data=payload, method="POST")
    auth = base64.b64encode(f"{sid}:{token}".encode("utf-8")).decode("ascii")
    request.add_header("Authorization", f"Basic {auth}")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Twilio WhatsApp send failed: HTTP {exc.code} - {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Twilio WhatsApp send failed: {exc.reason}") from exc


def send_low_risk_action_required(
    recipient_phone: str,
    amount_usd: float,
    transaction_date: str,
    account_last_4: str,
    channel: str,
) -> dict:
    """Send a customer-facing low-risk action-required WhatsApp message."""
    try:
        txn_dt = datetime.fromisoformat(str(transaction_date))
    except ValueError:
        try:
            txn_dt = datetime.strptime(str(transaction_date), "%Y-%m-%d")
        except ValueError:
            txn_dt = datetime.now()

    display_date = txn_dt.strftime("%d-%b-%y").upper()
    display_time = txn_dt.strftime("%I:%M:%S %p")

    message = (
        "⚠️ Transaction Alert!\n"
        "Immediate Action Required\n\n"
        f"Amount: USD {amount_usd:,.2f}\n"
        f"Account No.: xx{account_last_4}\n\n"
        f"Channel: {channel}\n"
        f"Date: {display_date}\n"
        f"Time: {display_time} (IST)\n\n"
        "Have you done this transaction?"
    )
    return send_whatsapp_message(to_number=recipient_phone, message=message)


def main() -> None:
    """Simple direct-run test for WhatsApp delivery."""
    print("=" * 60)
    print("  whatsapp_sender.py — self test")
    print("=" * 60)

    test_phone = WHATSAPP_TO_NUMBER or "+918446387628"
    print(f"Sending WhatsApp test message to: {test_phone}")

    try:
        response = send_low_risk_action_required(
            recipient_phone=test_phone,
            amount_usd=125.50,
            transaction_date="2026-04-11T10:06:14",
            account_last_4="7178",
            channel="iMobile",
        )
        print("WhatsApp message sent successfully")
        print(f"SID: {response.get('sid', '-')}")
        print(f"Status: {response.get('status', '-')}")
        print(f"To: {response.get('to', test_phone)}")
    except Exception as e:
        print(f"WhatsApp self test failed: {e}")


if __name__ == "__main__":
    main()
