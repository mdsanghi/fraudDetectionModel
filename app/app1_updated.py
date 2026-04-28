import streamlit as st
import pandas as pd
import sys
import base64
from html import escape
from datetime import date, datetime
from pathlib import Path

# Ensure the repository root is on sys.path so integration_layer imports work
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from app.url_qrGenerator import generate_streamlit_network_qr
except Exception as e:
    print(f"[app1_updated] QR generator import failed: {e}")
    generate_streamlit_network_qr = None

# Optional fraud integration
try:
    from integration_layer.fraud_bridge import run_fraud_check
except Exception as e:
    print(f"[app1_updated] fraud_bridge import failed: {e}")
    run_fraud_check = None

st.set_page_config(page_title="NBO Transfer", layout="centered", initial_sidebar_state="collapsed")

DATA_FILE = Path(__file__).parent.parent / "fraud_detection_synthetic_data.xlsx"
LOGO_FILE = Path(__file__).parent / "nbologo.jpeg"
BACKOFFICE_RELATIVE_URL = "?backoffice=1"

if generate_streamlit_network_qr is not None:
    try:
        QR_RESULT = generate_streamlit_network_qr()
        print(f"[app1_updated] Network URL QR generated: {QR_RESULT.qr_path} ({QR_RESULT.url})")
        print(f"[app1_updated] Stable back-office URL: {QR_RESULT.url}{BACKOFFICE_RELATIVE_URL}")
    except Exception as e:
        QR_RESULT = None
        print(f"[app1_updated] Network URL QR generation failed: {e}")
        print(f"[app1_updated] Stable back-office relative URL: {BACKOFFICE_RELATIVE_URL}")
else:
    QR_RESULT = None
    print(f"[app1_updated] Stable back-office relative URL: {BACKOFFICE_RELATIVE_URL}")


def image_to_data_uri(path: Path) -> str:
    mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


try:
    NBO_LOGO_DATA_URI = image_to_data_uri(LOGO_FILE)
except Exception as e:
    NBO_LOGO_DATA_URI = ""
    print(f"[app1_updated] logo load failed: {e}")


@st.cache_resource
def get_backoffice_state():
    return {"latest": None}


def get_backoffice_url() -> str:
    return BACKOFFICE_RELATIVE_URL


def publish_backoffice_review(transfer_data: dict, risk_result: dict) -> None:
    get_backoffice_state()["latest"] = {
        "transfer_data": dict(transfer_data),
        "risk_result": dict(risk_result),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    relative_url = get_backoffice_url()
    print("\n[app1_updated] BACK OFFICE REVIEW UPDATED")
    print(f"[app1_updated] Open manually: {relative_url}")
    if QR_RESULT is not None and getattr(QR_RESULT, "url", None):
        print(f"[app1_updated] Network back-office URL: {QR_RESULT.url}{relative_url}")
    print("[app1_updated] Mobile app flow remains on the customer receipt page.\n")


# ── Data Loading ───────────────────────────────────────────────────────────────
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


def parse_excel_dates(values) -> pd.Series:
    """Parse workbook dates stored as ISO text, Excel dates, or manual edits."""
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
        fallback = pd.to_datetime(normalized_text.where(remaining), errors="coerce")
        parsed.loc[fallback.notna()] = fallback[fallback.notna()]

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


@st.cache_data
def load_workbook(path_str: str):
    path = Path(path_str)
    customers = pd.read_excel(path, sheet_name="Customers")
    beneficiaries = pd.read_excel(path, sheet_name="Beneficiaries")
    transactions = pd.read_excel(path, sheet_name="Transactions")
    login_history = pd.read_excel(path, sheet_name="Login_History")

    # Normalize types
    for df in (customers, beneficiaries, transactions, login_history):
        if "customer_id" in df.columns:
            df["customer_id"] = df["customer_id"].astype(str).str.strip().str.upper()

    if "beneficiary_id" in beneficiaries.columns:
        beneficiaries["beneficiary_id"] = beneficiaries["beneficiary_id"].astype(str).str.strip()

    if "beneficiary_id" in transactions.columns:
        transactions["beneficiary_id"] = transactions["beneficiary_id"].astype(str).str.strip()

    if "date_of_birth" in customers.columns:
        customers["date_of_birth"] = parse_excel_dates(customers["date_of_birth"])
    if "registration_date" in customers.columns:
        customers["registration_date"] = parse_excel_dates(customers["registration_date"])
    if "added_date" in beneficiaries.columns:
        beneficiaries["added_date"] = parse_excel_dates(beneficiaries["added_date"])
    if "transaction_date" in transactions.columns:
        transactions["transaction_date"] = parse_excel_dates(transactions["transaction_date"])
    if "login_timestamp" in login_history.columns:
        login_history["login_timestamp"] = parse_excel_dates(login_history["login_timestamp"])

    return customers, beneficiaries, transactions, login_history


def make_account_number(customer_id: str) -> str:
    digits = "".join(ch for ch in str(customer_id) if ch.isdigit())
    digits = (digits + "10470376449016")[:14]
    return digits


def estimate_balance_omr(customer_row: pd.Series, transactions_df: pd.DataFrame) -> float:
    customer_id = str(customer_row["customer_id"]).strip().upper()
    txn_rows = transactions_df[transactions_df["customer_id"] == customer_id].copy()

    if "amount_usd" in txn_rows.columns and not txn_rows.empty:
        avg_usd = float(txn_rows["amount_usd"].mean())
        # POC estimate only
        return round(max(avg_usd * 0.35, 75.0), 3)

    typical = float(customer_row.get("typical_txn_amount", 1500) or 1500)
    return round(max(typical * 0.35, 75.0), 3)


def build_transfer_data(customer_row: pd.Series, beneficiary_row: pd.Series | None, transactions_df: pd.DataFrame):
    customer_name = str(customer_row.get("full_name", "Customer")).upper()
    preferred_channel = str(customer_row.get("preferred_channel", "Mobile App"))
    country = str(customer_row.get("country_of_residence", "Oman"))
    typical_txn = float(customer_row.get("typical_txn_amount", 15000) or 15000)
    primary_device = str(customer_row.get("primary_device", "iPhone 14"))

    account_number = make_account_number(customer_row.get("customer_id", "10470376449016"))
    balance = estimate_balance_omr(customer_row, transactions_df)

    if beneficiary_row is not None:
        beneficiary_id = str(beneficiary_row.get("beneficiary_id", ""))
        beneficiary_name = str(beneficiary_row.get("beneficiary_name", ""))
        beneficiary_account = str(beneficiary_row.get("account_number", ""))
        beneficiary_country = str(beneficiary_row.get("country", ""))
        beneficiary_bank = str(beneficiary_row.get("bank_name", ""))
        beneficiary_branch = str(beneficiary_row.get("country", ""))
        beneficiary_town = str(beneficiary_row.get("country", ""))
    else:
        beneficiary_id = ""
        beneficiary_name = ""
        beneficiary_account = ""
        beneficiary_country = ""
        beneficiary_bank = ""
        beneficiary_branch = ""
        beneficiary_town = ""

    return {
        "customer_id": str(customer_row.get("customer_id", "")),
        "customer_name": customer_name,
        "from_account": account_number,
        "from_account_name": f"{preferred_channel.upper()} ACCOUNT",
        "available_balance": balance,
        "beneficiary_id": beneficiary_id,
        "beneficiary_name": beneficiary_name,
        "beneficiary_account": beneficiary_account,
        "beneficiary_country": beneficiary_country,
        "beneficiary_bank": beneficiary_bank,
        "beneficiary_branch": beneficiary_branch,
        "beneficiary_town": beneficiary_town,
        "exchange_rate": 239.58 if beneficiary_country.lower() == "india" else 1.00,
        "amount_usd": round(max(typical_txn, 250.0), 2),
        "purpose": "Family maintenance",
        "remark": "",
        "transfer_type": "Transfer Immediately",
        "transfer_date": str(date.today()),
        "transaction_timestamp": datetime.now().isoformat(),
        "country_of_residence": country,
        "risk_tier": str(customer_row.get("risk_tier", "Low")),
        "kyc_status": str(customer_row.get("kyc_status", "Verified")),
        "account_status": str(customer_row.get("account_status", "Active")),
        "preferred_channel": preferred_channel,
        "channel": preferred_channel,
        "device_used": primary_device,
        "customer_phone": str(customer_row.get("phone", "")),
        "phone": str(customer_row.get("phone", "")),
        "customer_email": str(customer_row.get("email", "")),
        "email": str(customer_row.get("email", "")),
    }


def get_customer_bundle(customer_id: str, customers_df: pd.DataFrame, beneficiaries_df: pd.DataFrame, transactions_df: pd.DataFrame):
    cid = str(customer_id).strip().upper()
    customer_match = customers_df[customers_df["customer_id"] == cid]

    if customer_match.empty:
        return None

    customer_row = customer_match.iloc[0]
    customer_beneficiaries = beneficiaries_df[beneficiaries_df["customer_id"] == cid].copy()

    if not customer_beneficiaries.empty:
        customer_beneficiaries["added_date_dt"] = parse_excel_dates(customer_beneficiaries["added_date"])
        customer_beneficiaries = customer_beneficiaries.sort_values(["added_date_dt", "beneficiary_name"], ascending=[False, True])
        first_beneficiary = customer_beneficiaries.iloc[0]
    else:
        first_beneficiary = None

    transfer_data = build_transfer_data(customer_row, first_beneficiary, transactions_df)

    return {
        "customer_row": customer_row,
        "beneficiaries_df": customer_beneficiaries,
        "transfer_data": transfer_data,
    }


def build_beneficiary_cards(beneficiaries_df: pd.DataFrame):
    cards = []
    flag_map = {
        "india": "🇮🇳",
        "uae": "🇦🇪",
        "united arab emirates": "🇦🇪",
        "canada": "🇨🇦",
        "philippines": "🇵🇭",
        "pakistan": "🇵🇰",
        "bangladesh": "🇧🇩",
        "egypt": "🇪🇬",
        "saudi arabia": "🇸🇦",
        "united kingdom": "🇬🇧",
        "russia": "🇷🇺",
        "oman": "🇴🇲",
    }
    if beneficiaries_df is None or beneficiaries_df.empty:
        return cards

    for _, row in beneficiaries_df.iterrows():
        country = str(row.get("country", ""))
        flag = flag_map.get(country.strip().lower(), "🌍")
        cards.append({
            "beneficiary_id": str(row.get("beneficiary_id", "")),
            "name": str(row.get("beneficiary_name", "")),
            "account": str(row.get("account_number", "")),
            "bank": str(row.get("bank_name", "")),
            "country": country,
            "flag": flag,
            "is_verified": bool(row.get("is_verified", False)),
        })
    return cards


# ── Session State ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "screen": "login",
        "passcode": "",
        "customer_id_input": "",
        "selected_beneficiary": None,
        "otp": "",
        "current_customer": None,
        "customer_beneficiaries": [],
        "transfer_data": {
            "customer_id": "",
            "customer_name": "",
            "from_account": "",
            "from_account_name": "ACCOUNT",
            "available_balance": 0.0,
            "beneficiary_id": "",
            "beneficiary_name": "",
            "beneficiary_account": "",
            "beneficiary_country": "",
            "beneficiary_bank": "",
            "beneficiary_branch": "",
            "beneficiary_town": "",
            "exchange_rate": 239.58,
            "amount_usd": 1500.0,
            "purpose": "Family maintenance",
            "remark": "",
            "transfer_type": "Transfer Immediately",
            "transfer_date": str(date.today()),
            "transaction_timestamp": datetime.now().isoformat(),
            "country_of_residence": "",
            "risk_tier": "",
            "kyc_status": "",
            "account_status": "",
            "preferred_channel": "",
        "channel": "",
        "device_used": "",
        "customer_phone": "",
        "phone": "",
        "customer_email": "",
        "email": "",
    },
        "risk_result": None,
        "login_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

backoffice_param = st.query_params.get("backoffice", "")
if str(backoffice_param).lower() in {"1", "true", "yes"}:
    review = get_backoffice_state().get("latest")
    st.session_state.screen = "risk_review"
    st.session_state.backoffice_review_id = "latest"
    if review:
        st.session_state.transfer_data = dict(review["transfer_data"])
        st.session_state.risk_result = dict(review["risk_result"])
    else:
        st.session_state.backoffice_review_id = ""
        st.session_state.risk_result = None

# ── Load workbook once ─────────────────────────────────────────────────────────
try:
    CUSTOMERS_DF, BENEFICIARIES_DF, TRANSACTIONS_DF, LOGIN_HISTORY_DF = load_workbook(str(DATA_FILE))
    DATA_LOAD_ERROR = None
except Exception as e:
    CUSTOMERS_DF = BENEFICIARIES_DF = TRANSACTIONS_DF = LOGIN_HISTORY_DF = None
    DATA_LOAD_ERROR = str(e)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --app-shell-height: 100vh;
}

@supports (height: 100dvh) {
    :root {
        --app-shell-height: 100dvh;
    }
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp {
    background: #1456D8;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    min-height: var(--app-shell-height);
}

.block-container {
    max-width: 390px !important;
    min-height: var(--app-shell-height);
    height: var(--app-shell-height);
    padding: 0 !important;
    margin: 0 auto !important;
    margin-top: 24px !important;
    margin-bottom: 24px !important;
    background: #F2F5FC;
    position: relative;
    overflow-x: hidden;
    overflow-y: auto !important;
    border-radius: 36px !important;
    border: 8px solid #1a1a2e !important;
    box-shadow:
        0 0 0 2px #2a2a4a,
        0 32px 80px rgba(0, 0, 0, 0.45),
        inset 0 0 0 1px rgba(255,255,255,0.06) !important;
}

.block-container > div:first-child { border-radius: 28px 28px 0 0; overflow: hidden; }
.block-container > div:last-child  { border-radius: 0 0 28px 28px; overflow: hidden; }

.element-container, div[data-testid="stMarkdown"] {
    margin-bottom: 0 !important;
}
.main .block-container { padding-top: 0 !important; }
section[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }

div[data-testid="stTextInput"] > div,
div[data-testid="stNumberInput"] > div,
div[data-testid="stTextArea"] > div { background: transparent !important; }

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1.5px solid #D0D5E3 !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    font-size: 16px !important;
    box-shadow: none !important;
    caret-color: #1456D8;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #9CA3AF !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #1456D8 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(20,86,216,0.12) !important;
}

div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    color: #374151 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}

div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: #111827 !important;
    border: 1.5px solid #D0D5E3 !important;
    border-radius: 12px !important;
    min-height: 48px !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] span { color: #111827 !important; }

div[data-testid="stRadio"] > div { gap: 8px !important; }
div[data-testid="stRadio"] label {
    background: #fff;
    border: 1.5px solid #D0D5E3;
    border-radius: 10px;
    padding: 8px 14px !important;
    color: #111827 !important;
    font-size: 15px !important;
    font-weight: 400 !important;
}

.stButton > button {
    width: 100% !important;
    background: #1456D8 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50px !important;
    min-height: 50px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    box-shadow: 0 2px 8px rgba(20,86,216,0.25) !important;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.88 !important; }

.btn-outline .stButton > button {
    background: #FFFFFF !important;
    color: #1456D8 !important;
    border: 2px solid #1456D8 !important;
    box-shadow: none !important;
}

div[data-testid="stNumberInput"] button {
    background: #EEF2FA !important;
    color: #1456D8 !important;
    border-radius: 8px !important;
    min-height: 32px !important;
    border: 1px solid #D0D5E3 !important;
}

*:focus { outline: none !important; }

hr { border: none; border-top: 1px solid #E5E7EB; margin: 10px 0; }

div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 15px !important;
}

@media (max-width: 480px) {
    html, body, [data-testid="stAppViewContainer"], .stApp, section[data-testid="stMain"] {
        height: var(--app-shell-height);
        min-height: var(--app-shell-height);
    }

    section[data-testid="stMain"] > div:first-child {
        padding-top: 0 !important;
    }

    .block-container {
        max-width: 100vw !important;
        min-height: var(--app-shell-height) !important;
        height: var(--app-shell-height) !important;
        margin: 0 !important;
        border-radius: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }

    .block-container > div:first-child,
    .block-container > div:last-child {
        border-radius: 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)

if st.session_state.screen == "risk_review":
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp, section[data-testid="stMain"] {
        height: auto !important;
        min-height: 100vh !important;
        background: #F3F6FB !important;
    }
    .block-container {
        max-width: 1180px !important;
        width: min(1180px, calc(100vw - 40px)) !important;
        min-height: auto !important;
        height: auto !important;
        padding: 0 !important;
        margin: 24px auto !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        overflow: visible !important;
    }
    .block-container > div:first-child,
    .block-container > div:last-child {
        border-radius: 0 !important;
        overflow: visible !important;
    }
    @media (max-width: 480px) {
        .block-container {
            width: 100vw !important;
            margin: 0 !important;
            padding: 0 !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def go(screen):
    st.session_state.screen = screen
    st.rerun()

def steps_html(active):
    def cls(i): return "step-active" if i == active else "step-dot"
    return f"""
    <style>
    .steps-wrap {{ display:flex; align-items:center; justify-content:center; gap:0; padding:14px 0 6px; }}
    .step-dot, .step-active {{
        width:36px; height:36px; border-radius:50%; display:flex; align-items:center;
        justify-content:center; font-weight:700; font-size:16px;
    }}
    .step-dot {{ background:#D0D5E3; color:#fff; }}
    .step-active {{ background:#1456D8; color:#fff; box-shadow:0 2px 8px rgba(20,86,216,.35); }}
    .step-line {{ width:48px; height:3px; background:#D0D5E3; }}
    </style>
    <div class="steps-wrap">
      <span class="{cls(1)}">1</span>
      <span class="step-line"></span>
      <span class="{cls(2)}">2</span>
      <span class="step-line"></span>
      <span class="{cls(3)}">3</span>
    </div>"""

def top_bar(show_user=False):
    customer_name = st.session_state.transfer_data.get("customer_name", "")
    user_row = f'<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0 2px;font-size:15px;font-weight:500;color:rgba(255,255,255,.9);"><span>{customer_name}</span><span style="font-size:13px;opacity:.8;">★ Rewards</span></div>' if show_user and customer_name else ""
    logo_html = (
        f'<img src="{NBO_LOGO_DATA_URI}" alt="NBO" '
        'style="display:block;width:154px;height:auto;margin:0 auto 2px;" />'
        if NBO_LOGO_DATA_URI
        else '<div style="text-align:center;font-size:42px;font-weight:900;line-height:1.1;letter-spacing:1px;">NBO</div>'
    )
    st.markdown(f"""
    <div style="background:#1456D8;padding:14px 18px 18px;color:white;">
      {logo_html}
      {user_row}
    </div>
    """, unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div style="background:#DDE4F2;color:#1456D8;font-size:15px;font-weight:600;padding:11px 18px;border-bottom:1px solid #C8D0E4;">{text}</div>', unsafe_allow_html=True)

def footer_nav():
    st.markdown("""
    <div style="background:#1456D8;color:white;padding:12px 8px 18px;display:flex;
         justify-content:space-around;font-size:12px;font-weight:500;margin-top:12px;
         position:sticky;bottom:0;">
      <div style="text-align:center">📹<br>Videos</div>
      <div style="text-align:center">🎁<br>Offers</div>
      <div style="text-align:center">📷<br>Scan&Pay</div>
      <div style="text-align:center">🔔<br>Alerts</div>
      <div style="text-align:center">☰<br>More</div>
    </div>""", unsafe_allow_html=True)

def row_pair(label, value):
    return f'<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #F3F4F6;font-size:14px;"><span style="color:#6B7280;width:38%">{label}</span><span style="color:#111827;font-weight:500;width:60%;text-align:right">{value}</span></div>'

def blue_band(text):
    st.markdown(f'<div style="background:#1456D8;color:white;padding:9px 14px;font-size:14px;font-weight:600;border-radius:12px 12px 0 0;margin-top:10px;">{text}</div>', unsafe_allow_html=True)

RISK_STYLES = {
    "Low": ("✅", "#E7F6EF", "#12805C", "#C6EDD9"),
    "Medium": ("⚠️", "#FFF8E7", "#A96900", "#FFE6A0"),
    "High": ("🚨", "#FDECEC", "#B42318", "#FDBDBA"),
}

def render_success_card(td):
    st.markdown(f"""
    <div style="background:#E7F6EF;border-radius:14px;padding:14px;margin-bottom:14px;
                border:1px solid #C6EDD9;text-align:center;">
      <div style="font-size:28px;">✓</div>
      <div style="font-size:16px;font-weight:700;color:#12805C;margin-top:4px;">Transfer Submitted Successfully</div>
      <div style="font-size:13px;color:#374151;margin-top:4px;">USD {td['amount_usd']:,.2f} → {td['beneficiary_name']}</div>
    </div>""", unsafe_allow_html=True)

def render_risk_card(result):
    level = result["level"]
    icon, bg, fg, border = RISK_STYLES.get(level, RISK_STYLES["Low"])
    st.markdown(f"""
    <div style="background:{bg};border:1.5px solid {border};border-radius:14px;
                padding:14px 16px;margin-bottom:14px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <span style="font-size:22px;">{icon}</span>
        <span style="font-size:17px;font-weight:700;color:{fg};">{level} Suspicion</span>
        <span style="margin-left:auto;background:{border};color:{fg};font-size:13px;
                     font-weight:700;padding:3px 10px;border-radius:20px;">Score: {result['score']}</span>
      </div>
      {"".join(f'<div style="font-size:13px;color:{fg};margin-top:4px;">• {t}</div>' for t in result["triggered"])}
    </div>""", unsafe_allow_html=True)

def render_transfer_summary(td):
    blue_band("Transfer Summary")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;
                border-radius:0 0 14px 14px;padding:4px 14px 10px;margin-bottom:14px;">
      {row_pair("Customer", td.get('customer_name', ''))}
      {row_pair("Beneficiary", td['beneficiary_name'])}
      {row_pair("Amount", f"USD {td['amount_usd']:,.2f}")}
      {row_pair("Country", td['beneficiary_country'])}
      {row_pair("Purpose", td['purpose'])}
      {row_pair("Date", td['transfer_date'])}
    </div>""", unsafe_allow_html=True)

def render_customer_details(td):
    blue_band("Customer Details")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;
                border-radius:0 0 14px 14px;padding:4px 14px 10px;margin-bottom:14px;">
      {row_pair("Customer ID", td.get('customer_id', ''))}
      {row_pair("Name", td.get('customer_name', ''))}
      {row_pair("Account", td.get('from_account', ''))}
      {row_pair("Account Name", td.get('from_account_name', ''))}
      {row_pair("Risk Tier", td.get('risk_tier', ''))}
      {row_pair("KYC", td.get('kyc_status', ''))}
      {row_pair("Status", td.get('account_status', ''))}
      {row_pair("Residence", td.get('country_of_residence', ''))}
      {row_pair("Channel", td.get('channel', ''))}
      {row_pair("Device", td.get('device_used', ''))}
    </div>""", unsafe_allow_html=True)

def render_beneficiary_details(td):
    blue_band("Beneficiary Details")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;
                border-radius:0 0 14px 14px;padding:4px 14px 10px;margin-bottom:14px;">
      {row_pair("Beneficiary ID", td.get('beneficiary_id', ''))}
      {row_pair("Name", td.get('beneficiary_name', ''))}
      {row_pair("Account No.", td.get('beneficiary_account', ''))}
      {row_pair("Bank", td.get('beneficiary_bank', ''))}
      {row_pair("Country", td.get('beneficiary_country', ''))}
      {row_pair("Town", td.get('beneficiary_town', ''))}
    </div>""", unsafe_allow_html=True)

def compute_risk(data):
    score = 0
    triggered = []
    amount = float(data.get("amount_usd", 0))
    country = data.get("beneficiary_country", "").lower()
    purpose = data.get("purpose", "").lower()

    if amount >= 5000:
        score += 30
        triggered.append("High transfer amount (≥ USD 5,000)")
    elif amount >= 2000:
        score += 15
        triggered.append("Moderate transfer amount (≥ USD 2,000)")

    if country not in ["oman", "india", ""]:
        score += 35
        triggered.append("Less common beneficiary country")
    elif country == "india":
        score += 10
        triggered.append("Cross-border transfer to India")

    if purpose in ["gift", "personal", "others"]:
        score += 10
        triggered.append("Generic transfer purpose")

    if not str(data.get("remark", "")).strip():
        score += 5
        triggered.append("No remark provided")

    level = "High" if score >= 45 else "Medium" if score >= 20 else "Low"
    return {"score": score, "level": level, "triggered": triggered}


# SCREEN: LOGIN
if st.session_state.screen == "login":
    st.markdown("""
    <style>
    .stApp, body {
        background: linear-gradient(180deg, #1456D8 0%, #0F4CC2 100%) !important;
    }
    .block-container {
        background: linear-gradient(180deg, #1456D8 0 34%, #F7F9FE 34% 100%) !important;
        overflow-y: hidden !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div > div > [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div > div
      > [data-testid="stVerticalBlock"]
      > div:nth-child(n+3) {
        background: #F7F9FE;
        padding-left: 22px !important;
        padding-right: 22px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div > div
      > [data-testid="stVerticalBlock"]
      > div:nth-child(3) {
        border-radius: 30px 30px 0 0;
        padding-top: 18px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div > div
      > [data-testid="stVerticalBlock"]
      > div:last-child {
        border-radius: 0 0 30px 30px;
        padding-bottom: 20px !important;
    }
    div[data-testid="stTextInput"] {
        background: transparent !important;
        padding: 0 22px 8px !important;
    }
    div[data-testid="stTextInput"] > div {
        background: #FFFFFF !important;
        border: 1.5px solid #D7DEEE !important;
        border-radius: 16px !important;
        box-shadow: none !important;
        overflow: hidden !important;
    }
    div[data-testid="stTextInput"] > div:focus-within {
        border-color: #1456D8 !important;
        box-shadow: 0 0 0 4px rgba(20, 86, 216, 0.12) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stTextInput"] input {
        background: #FFFFFF !important;
        border: 1.5px solid #D7DEEE !important;
        border: none !important;
        border-radius: 16px !important;
        min-height: 56px !important;
        padding: 14px 18px !important;
        font-size: 17px !important;
        box-shadow: none !important;
        transition: box-shadow 0.18s ease, transform 0.18s ease !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border: none !important;
        box-shadow: none !important;
        transform: none !important;
    }
    div[data-testid="stTextInput"] button,
    div[data-testid="stTextInput"] [role="button"] {
        background: #FFFFFF !important;
        border: none !important;
        box-shadow: none !important;
        color: #374151 !important;
    }
    div[data-testid="stTextInput"] svg {
        fill: #374151 !important;
    }
    .stButton {
        background: transparent !important;
        padding: 6px 22px 0 !important;
    }
    .stButton > button {
        min-height: 54px !important;
        font-size: 18px !important;
        box-shadow: 0 14px 28px rgba(20, 86, 216, 0.22) !important;
    }
    div[data-testid="stAlert"] {
        margin: 8px 22px 0 !important;
    }
    .login-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.18);
        color: white;
        font-size: 13px;
        font-weight: 600;
    }
    .login-hint-card {
        background: linear-gradient(180deg, #FFFFFF 0%, #F1F5FE 100%);
        border: 1px solid #E4EAF7;
        border-radius: 18px;
        padding: 14px 16px;
        color: #41506A;
        text-align: center;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
    }
    .login-hint-card strong {
        color: #1456D8;
    }
    </style>
    """, unsafe_allow_html=True)

    login_logo_html = (
        f'<img src="{NBO_LOGO_DATA_URI}" alt="NBO" '
        'style="display:block;width:190px;height:auto;margin:0 auto 18px;" />'
        if NBO_LOGO_DATA_URI
        else '<div style="text-align:center;font-size:54px;font-weight:900;line-height:1;letter-spacing:2px;margin-bottom:16px;">NBO</div>'
    )

    st.markdown(f"""
    <div style="background:transparent;color:white;padding:20px 22px 18px;">
      {login_logo_html}
      <div style="font-size:21px;font-weight:700;margin-bottom:5px;">Welcome back</div>
      <div style="font-size:15px;opacity:.82;margin-bottom:12px;">Enter customer ID and passcode to continue</div>
      
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:transparent;padding:8px 22px 6px;
         margin-top:32px;border-top:0;">
      <p style="color:#374151;font-size:14px;font-weight:600;margin:0 0 8px;">Customer ID</p>
    </div>
    """, unsafe_allow_html=True)

    customer_id = st.text_input(
        "Customer ID",
        value=st.session_state.customer_id_input,
        placeholder="Enter customer id e.g. CUST1000",
        label_visibility="collapsed",
        key="customer_id_field"
    )
    st.session_state.customer_id_input = customer_id

    st.markdown('<div style="background:transparent;padding:4px 22px 6px;"><p style="color:#374151;font-size:14px;font-weight:600;margin:0 0 8px;">Passcode</p></div>', unsafe_allow_html=True)

    passcode = st.text_input(
        "Passcode",
        type="password",
        placeholder="Enter 4-digit passcode",
        label_visibility="collapsed",
        key="passcode_input"
    )
    st.session_state.passcode = passcode

    if DATA_LOAD_ERROR:
        st.error(f"Unable to load workbook: {DATA_LOAD_ERROR}")

    st.markdown('<div style="background:transparent;padding:4px 22px 0;"></div>', unsafe_allow_html=True)

    if st.button("Login", key="login_btn"):
        cid = customer_id.strip().upper()

        if DATA_LOAD_ERROR:
            st.session_state.login_error = f"Workbook error: {DATA_LOAD_ERROR}"
        elif not cid:
            st.session_state.login_error = "Please enter a customer ID."
        else:
            bundle = get_customer_bundle(cid, CUSTOMERS_DF, BENEFICIARIES_DF, TRANSACTIONS_DF)
            if bundle is None:
                st.session_state.login_error = f"No customer found for ID: {cid}"
            else:
                st.session_state.current_customer = cid
                st.session_state.transfer_data = bundle["transfer_data"]
                st.session_state.customer_beneficiaries = build_beneficiary_cards(bundle["beneficiaries_df"])
                st.session_state.selected_beneficiary = st.session_state.transfer_data.get("beneficiary_name")
                st.session_state.login_error = ""
                go("dashboard")

    if st.session_state.login_error:
        st.error(st.session_state.login_error)

    st.markdown("""
    <div style="background:transparent;padding:14px 22px 20px;">
      <div class="login-hint-card">
        Try a valid ID like <strong>CUST1000</strong><br>
        Customer data will be loaded directly from Excel.
      </div>
    </div>
    """, unsafe_allow_html=True)


elif st.session_state.screen == "dashboard":
    top_bar(show_user=True)
    section_title("Fund Transfer")
    st.markdown('<div style="padding:8px 0;">', unsafe_allow_html=True)

    items = [
        ("Within My Accounts", "Between any of your NBO Accounts", True),
        ("Instant Transfer", "India · UAE · Pakistan · Bangladesh · Philippines · Egypt", True),
        ("To Other Accounts", "NBO accounts, local banks & worldwide", True),
        ("Manage Beneficiaries", "Add, edit or remove beneficiaries", True),
        ("Mobile Payment", "Wallet transfer within NBO", True),
        ("Scheduled Transfers", "View or cancel pending transfers", True),
        ("Transfer History", "All past online transfers", True),
        ("Remittance Alert", "Set up SMS alert for currency rates", True),
    ]

    st.markdown("""<style>
    div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        align-items: stretch !important;
        background: white;
        border-bottom: 1px solid #F3F4F6;
    }
    div[data-testid="stHorizontalBlock"] > div:first-child {
        flex: 1 !important;
        padding: 0 !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child {
        flex: 0 0 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        padding: 0 12px 0 0 !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child .stButton > button {
        background: transparent !important;
        color: #1456D8 !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 22px !important;
        font-weight: 400 !important;
        padding: 0 !important;
        min-height: unset !important;
        width: auto !important;
        line-height: 1 !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child .stButton > button:hover {
        opacity: 0.7 !important;
        background: transparent !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stMarkdown"] {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
    }
    </style>""", unsafe_allow_html=True)

    for idx, (title, sub, clickable) in enumerate(items):
        col_a, col_b = st.columns([9, 1])
        with col_a:
            st.markdown(f"""
            <div style="padding:14px 16px;">
              <div style="font-size:15px;font-weight:600;color:#111827;">{title}</div>
              <div style="font-size:13px;color:#6B7280;margin-top:3px;">{sub}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            if clickable:
                if st.button("›", key=f"dash_{idx}"):
                    go("beneficiary")
            else:
                st.markdown("<div style='font-size:22px;color:#1456D8;'>›</div>", unsafe_allow_html=True)

    footer_nav()


elif st.session_state.screen == "beneficiary":
    top_bar(show_user=True)
    section_title("Select Beneficiary")
    st.markdown('<div style="padding:14px 16px;">', unsafe_allow_html=True)
    st.markdown(steps_html(1), unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    benefs = st.session_state.customer_beneficiaries or []

    search = st.text_input(
        "Search",
        placeholder="🔍  Search name, bank or country",
        label_visibility="collapsed"
    )

    if search:
        s = search.lower()
        benefs = [b for b in benefs if s in (b["name"] + b["bank"] + b["country"]).lower()]

    if not benefs:
        st.info("No beneficiaries found for this customer.")

    # Match the dashboard UI style for horizontal column blocks
    st.markdown("""<style>
    div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        align-items: stretch !important;
        background: white;
        border-bottom: 1px solid #F3F4F6;
    }
    div[data-testid="stHorizontalBlock"] > div:first-child {
        flex: 1 !important;
        padding: 0 !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child {
        flex: 0 0 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        padding: 0 12px 0 0 !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child .stButton > button {
        background: transparent !important;
        color: #1456D8 !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 22px !important;
        font-weight: 400 !important;
        padding: 0 !important;
        min-height: unset !important;
        width: auto !important;
        line-height: 1 !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child .stButton > button:hover {
        opacity: 0.7 !important;
        background: transparent !important;
    }
    div[data-testid="stHorizontalBlock"] > div:last-child div[data-testid="stMarkdown"] {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
    }
    </style>""", unsafe_allow_html=True)

    for i, b in enumerate(benefs):
        verified_txt = "Verified" if b.get("is_verified") else "Pending verification"
        col_a, col_b = st.columns([9, 1])
        with col_a:
            # Inline background and border removed; UI now relies on the global row CSS
            st.markdown(f"""
            <div style="padding:14px 16px;">
              <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:4px;">{b['flag']}&nbsp;&nbsp;{b['name']}</div>
              <div style="font-size:12px;color:#6B7280;margin-bottom:3px;">{b['account']}</div>
              <div style="font-size:12px;color:#9CA3AF;">{b['bank']} · {b['country']} · {verified_txt}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            if st.button("›", key=f"sel_{i}"):
                td = st.session_state.transfer_data
                td["beneficiary_id"] = b["beneficiary_id"]
                td["beneficiary_name"] = b["name"]
                td["beneficiary_account"] = b["account"]
                td["beneficiary_bank"] = b["bank"]
                td["beneficiary_country"] = b["country"]
                td["beneficiary_branch"] = b["country"]
                td["beneficiary_town"] = b["country"]
                td["exchange_rate"] = 239.58 if b["country"].strip().lower() == "india" else 1.00
                st.session_state.selected_beneficiary = b["name"]
                go("details")

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
    if st.button("← Back", key="bene_back"):
        go("dashboard")
    st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()


elif st.session_state.screen == "details":
    td = st.session_state.transfer_data
    top_bar(show_user=True)
    section_title("Transfer Details")
    st.markdown('<div style="padding:14px 16px;">', unsafe_allow_html=True)
    st.markdown(steps_html(2), unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#EEF2FA;border-radius:14px;padding:12px 14px;margin-bottom:14px;
                border-left:4px solid #1456D8;">
      <div style="font-size:12px;color:#6B7280;margin-bottom:3px;">From Account</div>
      <div style="font-size:15px;font-weight:600;color:#111827;">{td['from_account']}</div>
      <div style="font-size:13px;color:#374151;">{td['from_account_name']}</div>
      <div style="font-size:13px;color:#1456D8;font-weight:500;margin-top:4px;">Balance: USD {td['available_balance']:.2f}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#EEF2FA;border-radius:14px;padding:12px 14px;margin-bottom:16px;
                border-left:4px solid #12805C;">
      <div style="font-size:12px;color:#6B7280;margin-bottom:3px;">To Beneficiary</div>
      <div style="font-size:15px;font-weight:600;color:#111827;">{td['beneficiary_name']}</div>
      <div style="font-size:13px;color:#374151;">{td['beneficiary_bank']} · {td['beneficiary_country']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="background:#FFF8E7;border-radius:10px;padding:9px 12px;font-size:13px;color:#7D4E00;margin-bottom:14px;">Fraud checks use USD only. The OMR figure below is just a display equivalent.</div>', unsafe_allow_html=True)

    st.markdown('<p style="color:#374151;font-size:14px;font-weight:500;margin-bottom:4px;">Amount (USD)</p>', unsafe_allow_html=True)
    amount = st.number_input("amount_lbl", min_value=0.0, value=float(td["amount_usd"]), step=100.0, label_visibility="collapsed")
    td["amount_usd"] = amount
    td["transaction_timestamp"] = datetime.now().isoformat()

    omr_equiv = amount / 2.60 if 2.60 else amount
    st.markdown(f'<div style="font-size:13px;color:#6B7280;margin-top:-8px;margin-bottom:12px;">Display equivalent: ≈ OMR {omr_equiv:.3f}</div>', unsafe_allow_html=True)

    st.markdown('<p style="color:#374151;font-size:14px;font-weight:500;margin-bottom:4px;">Transfer Timing</p>', unsafe_allow_html=True)
    transfer_type = st.radio(
        "timing_lbl",
        ["Transfer Immediately", "Select Future Date"],
        index=0 if td["transfer_type"] == "Transfer Immediately" else 1,
        label_visibility="collapsed"
    )
    td["transfer_type"] = transfer_type
    if transfer_type == "Select Future Date":
        td["transfer_date"] = str(st.date_input("Transfer date", value=date.today()))
    else:
        td["transfer_date"] = str(date.today())

    st.markdown('<p style="color:#374151;font-size:14px;font-weight:500;margin:10px 0 4px;">Purpose of Remittance</p>', unsafe_allow_html=True)
    purposes = ["Family maintenance", "Education", "Medical", "Gift", "Personal", "Others"]
    td["purpose"] = st.selectbox(
        "purpose_lbl",
        purposes,
        index=purposes.index(td["purpose"]) if td["purpose"] in purposes else 0,
        label_visibility="collapsed"
    )

    st.markdown('<p style="color:#374151;font-size:14px;font-weight:500;margin:10px 0 4px;">Remark (optional)</p>', unsafe_allow_html=True)
    td["remark"] = st.text_input("remark_lbl", value=td["remark"], placeholder="e.g. Monthly support", label_visibility="collapsed")

    st.markdown('<p style="color:#374151;font-size:14px;font-weight:500;margin:10px 0 4px;">Beneficiary Town</p>', unsafe_allow_html=True)
    td["beneficiary_town"] = st.text_input("town_lbl", value=td["beneficiary_town"], label_visibility="collapsed")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="btn-outline" style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("← Back", key="det_back"):
            go("beneficiary")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("Next →", key="det_next"):
            go("confirm")
        st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()


elif st.session_state.screen == "confirm":
    td = st.session_state.transfer_data
    omr_debit = td["amount_usd"] / 2.60 if 2.60 else td["amount_usd"]
    top_bar(show_user=True)
    section_title("Confirm Transaction")
    st.markdown('<div style="padding:14px 16px;">', unsafe_allow_html=True)
    st.markdown(steps_html(3), unsafe_allow_html=True)
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    blue_band("Transaction Details")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;border-radius:0 0 14px 14px;padding:4px 14px 4px;">
      {row_pair("Customer ID", td.get('customer_id', ''))}
      {row_pair("From", td['from_account'])}
      {row_pair("Account", td['from_account_name'])}
      {row_pair("To", td['beneficiary_account'])}
    </div>""", unsafe_allow_html=True)

    blue_band("Transfer Amount")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;border-radius:0 0 14px 14px;padding:4px 14px 4px;">
      {row_pair("Amount", f"USD {td['amount_usd']:,.2f}")}
      {row_pair("Service Charges", "USD 0.00")}
      {row_pair("Debit Amount", f"USD {td['amount_usd']:,.2f}")}
    </div>""", unsafe_allow_html=True)

    blue_band("Beneficiary Details")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;border-radius:0 0 14px 14px;padding:4px 14px 4px;">
      {row_pair("Name", td['beneficiary_name'])}
      {row_pair("Account No.", td['beneficiary_account'])}
      {row_pair("Bank", td['beneficiary_bank'])}
      {row_pair("Branch", td['beneficiary_branch'])}
      {row_pair("Country", td['beneficiary_country'])}
    </div>""", unsafe_allow_html=True)

    blue_band("Customer Profile")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;border-radius:0 0 14px 14px;padding:4px 14px 4px;">
      {row_pair("Risk Tier", td.get('risk_tier', ''))}
      {row_pair("KYC", td.get('kyc_status', ''))}
      {row_pair("Status", td.get('account_status', ''))}
      {row_pair("Residence", td.get('country_of_residence', ''))}
    </div>""", unsafe_allow_html=True)

    blue_band("Schedule")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;border-radius:0 0 14px 14px;padding:4px 14px 14px;">
      {row_pair("Type", td['transfer_type'])}
      {row_pair("Purpose", td['purpose'])}
      {row_pair("Date", td['transfer_date'])}
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="btn-outline" style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("← Back", key="conf_back"):
            go("details")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("Confirm ✓", key="conf_next"):
            go("otp")
        st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()


elif st.session_state.screen == "otp":
    top_bar(show_user=True)
    section_title("OTP Verification")
    st.markdown('<div style="padding:24px 20px;">', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border-radius:20px;padding:24px 20px;text-align:center;
                border:1px solid #E5E7EB;margin-bottom:20px;">
      <div style="font-size:36px;margin-bottom:12px;">🔐</div>
      <div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:6px;">Enter OTP</div>
      <div style="font-size:14px;color:#6B7280;">A 6-digit OTP has been sent to your registered mobile number</div>
    </div>""", unsafe_allow_html=True)

    otp = st.text_input("otp_lbl", type="password", max_chars=6, placeholder="• • • • • •", label_visibility="collapsed")
    st.session_state.otp = otp

    st.markdown('<div style="text-align:center;color:#1456D8;font-size:14px;font-weight:500;margin:12px 0 18px;">Resend OTP</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="btn-outline" style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("← Back", key="otp_back"):
            go("confirm")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("Submit OTP", key="otp_submit"):
            with st.spinner("Verifying transaction..."):
                print("[app1_updated] run_fraud_check available:", run_fraud_check is not None)
                if run_fraud_check is not None:
                    print("[app1_updated] TRANSFER DATA:", st.session_state.transfer_data)
                    st.session_state.risk_result = run_fraud_check(st.session_state.transfer_data)
                    print("[app1_updated] RISK RESULT:", st.session_state.risk_result)
                else:
                    st.session_state.risk_result = compute_risk(st.session_state.transfer_data)
            publish_backoffice_review(
                st.session_state.transfer_data,
                st.session_state.risk_result,
            )
            go("result")
        st.markdown('</div>', unsafe_allow_html=True)

    footer_nav()


elif st.session_state.screen == "risk_review":
    latest_review = get_backoffice_state().get("latest")
    if not latest_review:
        st.markdown("""
        <div style="background:#F8FAFD;border:1px solid #DDE5F2;border-radius:10px;
                    padding:24px;color:#111827;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div style="font-size:22px;font-weight:800;margin-bottom:8px;">Back Office Transaction Review</div>
          <div style="font-size:14px;color:#667085;">
            No transaction has been submitted since this Streamlit program launched.
            Keep this URL open and refresh it after a transfer is completed.
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    td = st.session_state.transfer_data
    result = st.session_state.risk_result or compute_risk(td)
    level = result.get("level", "Low")
    _, _, risk_color, risk_border = RISK_STYLES.get(level, RISK_STYLES["Low"])

    def val(key, fallback=""):
        return escape(str(td.get(key, fallback) or ""))

    def metric(label, value):
        return (
            f'<div class="bo-metric">'
            f'<div class="bo-metric-label">{escape(str(label))}</div>'
            f'<div class="bo-metric-value">{escape(str(value))}</div>'
            f'</div>'
        )

    def detail_row(label, value):
        return (
            f'<div class="bo-row">'
            f'<span>{escape(str(label))}</span>'
            f'<strong>{escape(str(value or ""))}</strong>'
            f'</div>'
        )

    rules_html = "".join(
        f'<li>{escape(str(rule))}</li>'
        for rule in result.get("triggered", [])
    )
    metrics_html = "".join([
        metric("Risk Level", level),
        metric("Risk Score", result.get("score", "")),
        metric("Amount", f"USD {float(td.get('amount_usd', 0)):,.2f}"),
        metric("Status", "Latest Transaction"),
    ])
    transaction_rows_html = "".join([
        detail_row("Customer Name", td.get("customer_name", "")),
        detail_row("From Account", td.get("from_account", "")),
        detail_row("Account Name", td.get("from_account_name", "")),
        detail_row("Beneficiary Name", td.get("beneficiary_name", "")),
        detail_row("Beneficiary Account", td.get("beneficiary_account", "")),
        detail_row("Beneficiary Bank", td.get("beneficiary_bank", "")),
        detail_row("Beneficiary Country", td.get("beneficiary_country", "")),
        detail_row("Amount", f"USD {float(td.get('amount_usd', 0)):,.2f}"),
        detail_row("Purpose", td.get("purpose", "")),
        detail_row("Transfer Type", td.get("transfer_type", "")),
        detail_row("Transfer Date", td.get("transfer_date", "")),
    ])
    customer_rows_html = "".join([
        detail_row("Customer ID", td.get("customer_id", "")),
        detail_row("Name", td.get("customer_name", "")),
        detail_row("Risk Tier", td.get("risk_tier", "")),
        detail_row("KYC Status", td.get("kyc_status", "")),
        detail_row("Account Status", td.get("account_status", "")),
        detail_row("Residence", td.get("country_of_residence", "")),
        detail_row("Phone", td.get("customer_phone") or td.get("phone", "")),
        detail_row("Email", td.get("customer_email") or td.get("email", "")),
        detail_row("Preferred Channel", td.get("preferred_channel", "")),
        detail_row("Transaction Channel", td.get("channel", "")),
        detail_row("Device Used", td.get("device_used", "")),
    ])
    beneficiary_rows_html = "".join([
        detail_row("Beneficiary ID", td.get("beneficiary_id", "")),
        detail_row("Name", td.get("beneficiary_name", "")),
        detail_row("Bank", td.get("beneficiary_bank", "")),
        detail_row("Country", td.get("beneficiary_country", "")),
        detail_row("Town", td.get("beneficiary_town", "")),
    ])

    st.markdown("""
    <style>
    .bo-page {
        color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        padding-bottom: 24px;
    }
    .bo-topbar {
        background: #0F2748;
        color: white;
        padding: 18px 22px;
        border-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 18px;
        margin-bottom: 16px;
    }
    .bo-kicker {
        color: #93B4E8;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 5px;
    }
    .bo-title {
        font-size: 24px;
        font-weight: 800;
        line-height: 1.2;
    }
    .bo-subtitle {
        font-size: 14px;
        color: rgba(255,255,255,0.76);
        margin-top: 5px;
    }
    .bo-ref {
        text-align: right;
        color: rgba(255,255,255,0.78);
        font-size: 12px;
        line-height: 1.7;
    }
    .bo-metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }
    .bo-metric {
        background: white;
        border: 1px solid #DDE5F2;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 78px;
    }
    .bo-metric-label {
        color: #667085;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .bo-metric-value {
        color: #111827;
        font-size: 20px;
        font-weight: 800;
        line-height: 1.2;
    }
    .bo-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(340px, 0.8fr);
        gap: 16px;
        align-items: start;
    }
    .bo-panel {
        background: white;
        border: 1px solid #DDE5F2;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 16px;
    }
    .bo-panel-title {
        background: #F8FAFD;
        border-bottom: 1px solid #E6ECF5;
        color: #1F2937;
        font-size: 14px;
        font-weight: 800;
        padding: 12px 14px;
    }
    .bo-panel-body {
        padding: 6px 14px 12px;
    }
    .bo-row {
        display: grid;
        grid-template-columns: 190px minmax(0, 1fr);
        gap: 14px;
        padding: 10px 0;
        border-bottom: 1px solid #F0F3F8;
        font-size: 13px;
    }
    .bo-row:last-child {
        border-bottom: 0;
    }
    .bo-row span {
        color: #667085;
        font-weight: 600;
    }
    .bo-row strong {
        color: #111827;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    .bo-risk-panel {
        border-top: 5px solid var(--risk-color);
    }
    .bo-score {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 16px 14px;
        border-bottom: 1px solid #F0F3F8;
    }
    .bo-score-name {
        color: var(--risk-color);
        font-size: 22px;
        font-weight: 900;
    }
    .bo-score-pill {
        background: var(--risk-bg);
        color: var(--risk-color);
        border: 1px solid var(--risk-border);
        border-radius: 999px;
        padding: 7px 13px;
        font-size: 14px;
        font-weight: 900;
        white-space: nowrap;
    }
    .bo-rule-list {
        margin: 0;
        padding: 6px 18px 16px 34px;
        color: #B42318;
        font-size: 14px;
        line-height: 1.55;
    }
    .bo-rule-list li {
        margin: 8px 0;
    }
    .bo-actions {
        max-width: 360px;
        margin-top: 4px;
    }
    @media (max-width: 820px) {
        .bo-topbar {
            display: block;
        }
        .bo-ref {
            text-align: left;
            margin-top: 12px;
        }
        .bo-metrics,
        .bo-grid,
        .bo-actions {
            grid-template-columns: 1fr;
        }
        .bo-row {
            grid-template-columns: 1fr;
            gap: 3px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.html(f"""
    <div class="bo-page" style="--risk-color:{risk_color};--risk-border:{risk_border};--risk-bg:{risk_border}55;">
      <div class="bo-topbar">
        <div>
          <div class="bo-kicker">Back Office Fraud Operations</div>
          <div class="bo-title">Transaction Risk Review</div>
          <div class="bo-subtitle">Shows the latest submitted transfer from this running Streamlit program.</div>
        </div>
        <div class="bo-ref">
          <div>Customer: {val('customer_id')}</div>
          <div>Beneficiary: {val('beneficiary_id')}</div>
          <div>Date: {val('transfer_date')}</div>
        </div>
      </div>

      <div class="bo-metrics">
        {metrics_html}
      </div>

      <div class="bo-grid">
        <div>
          <div class="bo-panel bo-risk-panel">
            <div class="bo-panel-title">Risk Score and Triggered Rules</div>
            <div class="bo-score">
              <div class="bo-score-name">{escape(level)} Suspicion</div>
              <div class="bo-score-pill">Score: {escape(str(result.get("score", "")))}</div>
            </div>
            <ul class="bo-rule-list">{rules_html}</ul>
          </div>

          <div class="bo-panel">
            <div class="bo-panel-title">Transaction Details</div>
            <div class="bo-panel-body">
              {transaction_rows_html}
            </div>
          </div>
        </div>

        <div>
          <div class="bo-panel">
            <div class="bo-panel-title">Customer Profile</div>
            <div class="bo-panel-body">
              {customer_rows_html}
            </div>
          </div>

          <div class="bo-panel">
            <div class="bo-panel-title">Beneficiary Profile</div>
            <div class="bo-panel-body">
              {beneficiary_rows_html}
            </div>
          </div>
        </div>
      </div>
    </div>
    """)

    st.markdown('<div class="bo-actions">', unsafe_allow_html=True)
    if st.button("Call Customer", key="risk_call_customer"):
        customer_phone = td.get("customer_phone") or td.get("phone") or "phone number unavailable"
        customer_name = td.get("customer_name") or td.get("customer_id") or "customer"
        st.session_state.backoffice_call_message = f"Call requested for {customer_name}: {customer_phone}"
        print(f"[app1_updated] Back-office call requested: {customer_name} | {customer_phone}")
    if st.session_state.get("backoffice_call_message"):
        st.success(st.session_state.backoffice_call_message)
    st.markdown('</div>', unsafe_allow_html=True)


elif st.session_state.screen == "result":
    td = st.session_state.transfer_data
    result = st.session_state.risk_result or compute_risk(td)
    level = result["level"]

    top_bar(show_user=True)
    section_title("Transfer Initiated")
    st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)

    render_success_card(td)

    # Manual toggle: comment this line to hide the mobile risk score and triggered rules.
    render_risk_card(result)

    # if level in ("Medium", "High"):
    #     st.markdown("""
    #     <div style="background:#FFF8E7;border:1px solid #FFE6A0;border-radius:14px;
    #                 padding:12px 14px;margin-bottom:14px;color:#7D4E00;font-size:13px;">
    #       The stable back-office review URL has been printed in the Streamlit logs.
    #     </div>""", unsafe_allow_html=True)

    render_transfer_summary(td)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="btn-outline" style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("New Transfer", key="res_new"):
            st.session_state.risk_result = None
            go("dashboard")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="display:flex;align-items:stretch;">', unsafe_allow_html=True)
        if st.button("Restart Demo", key="res_restart"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    footer_nav()
