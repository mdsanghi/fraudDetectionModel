import streamlit as st
import pandas as pd
from datetime import date
from integration_layer.fraud_bridge import run_fraud_check
st.set_page_config(page_title="NBO Transfer", layout="centered", initial_sidebar_state="collapsed")

# ── Session State ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "screen": "login",
        "passcode": "",
        "selected_beneficiary": None,
        "otp": "",
        "transfer_data": {
            "customer_name": "MAYANK SANGHI",
            "from_account": "10470376449016",
            "from_account_name": "STAFF SAVINGS ACCOUNT OMR",
            "available_balance": 152.877,
            "beneficiary_name": "deepak india",
            "beneficiary_account": "37445612357",
            "beneficiary_country": "India",
            "beneficiary_bank": "STATE BANK OF INDIA",
            "beneficiary_branch": "JAYSINGPUR",
            "beneficiary_town": "Jaysingpur",
            "exchange_rate": 239.58,
            "amount_inr": 15000.0,
            "purpose": "Family maintenance",
            "remark": "",
            "transfer_type": "Transfer Immediately",
            "transfer_date": str(date.today()),
        },
        "risk_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp {
    background: #1456D8;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Hidden beneficiary trigger button */
.bene-hidden-btn {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
.bene-hidden-btn .stButton,
.bene-hidden-btn .stButton > button,
.bene-hidden-btn button {
    width: 1px !important;
    height: 1px !important;
    min-height: 1px !important;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    opacity: 0 !important;
    background: transparent !important;
    color: transparent !important;
    box-shadow: none !important;
    pointer-events: none !important;
}

/* Clickable beneficiary card */
.bene-card {
    background: white;
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
    border: 1px solid #E5E7EB;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.18s ease;
}
.bene-card:hover {
    border-color: #1456D8;
    box-shadow: 0 4px 14px rgba(20, 86, 216, 0.10);
    transform: translateY(-1px);
}
.bene-card-title {
    font-size: 15px;
    font-weight: 600;
    color: #111827;
}
.bene-card-sub {
    font-size: 12px;
    color: #6B7280;
    margin-top: 3px;
}
.bene-card-bank {
    font-size: 12px;
    color: #9CA3AF;
}
.bene-chevron {
    color: #1456D8;
    font-size: 22px;
    font-weight: 700;
    flex-shrink: 0;
    margin-left: 8px;
}

.block-container {
    max-width: 390px !important;
    min-height: 100vh;
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

def top_bar(title="", show_user=False):
    user_row = f'<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0 2px;font-size:15px;font-weight:500;color:rgba(255,255,255,.9);"><span>{st.session_state.transfer_data["customer_name"]}</span><span style="font-size:13px;opacity:.8;">★ Rewards</span></div>' if show_user else ""
    st.markdown(f"""
    <div style="background:#1456D8;padding:14px 18px 18px;color:white;">
      <div style="display:flex;justify-content:space-between;font-size:13px;font-weight:600;margin-bottom:10px;opacity:.9;">
        <span>9:41</span><span>📶 WiFi 🔋 88%</span>
      </div>
      <div style="text-align:center;font-size:13px;font-weight:600;opacity:.85;letter-spacing:.5px;">البنك الوطني العماني</div>
      <div style="text-align:center;font-size:42px;font-weight:900;line-height:1.1;letter-spacing:1px;">NBO</div>
      {user_row}
    </div>
    """, unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div style="background:#DDE4F2;color:#1456D8;font-size:15px;font-weight:600;padding:11px 18px;border-bottom:1px solid #C8D0E4;">{text}</div>', unsafe_allow_html=True)

def card(content_html):
    st.markdown(f'<div style="background:#fff;border:1px solid #E5E7EB;border-radius:16px;padding:14px 16px;margin-bottom:12px;">{content_html}</div>', unsafe_allow_html=True)

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

# ── Risk Engine ────────────────────────────────────────────────────────────────
def compute_risk(data):
    score = 0
    triggered = []
    amount = float(data.get("amount_inr", 0))
    country = data.get("beneficiary_country", "").lower()
    purpose = data.get("purpose", "").lower()

    if amount >= 20000:
        score += 30
        triggered.append("High transfer amount (≥ INR 20,000)")
    elif amount >= 10000:
        score += 15
        triggered.append("Moderate transfer amount (≥ INR 10,000)")

    if country not in ["oman", "india"]:
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

# ════════════════════════════════════════════════════════════════════════════════
# SCREEN: LOGIN
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "login":
    st.markdown("""
    <style>
    .stApp, body, .block-container {
        background: #1456D8 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div > div
      > [data-testid="stVerticalBlock"]
      > div:nth-child(n+4) {
        background: white;
        padding-left: 22px !important;
        padding-right: 22px !important;
    }
    div[data-testid="stTextInput"] { background: white !important; padding: 0 22px !important; }
    .stButton { background: white !important; padding: 0 22px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1456D8;color:white;padding:18px 22px 24px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;
           font-weight:600;opacity:.85;margin-bottom:18px;">
        <span>9:41</span><span>📶 WiFi 🔋 88%</span>
      </div>
      <div style="text-align:center;font-size:13px;font-weight:600;opacity:.8;
           letter-spacing:.5px;margin-bottom:5px;">البنك الوطني العماني</div>
      <div style="text-align:center;font-size:54px;font-weight:900;line-height:1;
           letter-spacing:2px;margin-bottom:28px;">NBO</div>
      <div style="font-size:21px;font-weight:700;margin-bottom:5px;">Welcome back 👋</div>
      <div style="font-size:15px;opacity:.82;margin-bottom:0;">Enter your passcode to continue</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border-radius:28px 28px 0 0;padding:28px 22px 6px;
         margin-top:0;border-top:0;">
      <p style="color:#374151;font-size:14px;font-weight:600;margin:0 0 8px;">Passcode</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stTextInput"] {
        background: white !important;
        padding: 0 22px 0 !important;
        margin: 0 !important;
    }
    div[data-testid="element-container"]:has(div[data-testid="stTextInput"]),
    div[data-testid="element-container"]:has(.stButton),
    div[data-testid="element-container"]:has(div[data-testid="stMarkdown"]:not(:first-child)) {
        background: white !important;
    }
    div[data-testid="stMarkdown"] + div[data-testid="element-container"] {
        margin-top: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    passcode = st.text_input(
        "Passcode",
        type="password",
        placeholder="Enter 4-digit passcode",
        label_visibility="collapsed",
        key="passcode_input"
    )
    st.session_state.passcode = passcode

    st.markdown('<div style="background:white;padding:12px 22px 0;margin-bottom:14px;"></div>', unsafe_allow_html=True)
    if st.button("Login", key="login_btn"):
        go("dashboard")

    st.markdown("""
    <div style="background:white;padding:20px 22px 32px;min-height:30vh;">
      <div style="text-align:center;color:#9CA3AF;font-size:14px;margin-bottom:14px;">
        ── or ──</div>
      <div style="text-align:center;font-size:44px;line-height:1.1;">🙂</div>
      <div style="text-align:center;color:#374151;font-size:15px;
           font-weight:600;margin:8px 0 14px;">Face ID</div>
      <div style="text-align:center;">
        <span style="color:#1456D8;font-size:14px;font-weight:500;">
          New User? Register here</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SCREEN: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "dashboard":
    top_bar(show_user=True)
    section_title("Fund Transfer")
    st.markdown('<div style="padding:8px 0;">', unsafe_allow_html=True)

    items = [
        ("Within My Accounts", "Between any of your NBO Accounts", False),
        ("Instant Transfer", "India · UAE · Pakistan · Bangladesh · Philippines · Egypt", True),
        ("To Other Accounts", "NBO accounts, local banks & worldwide", False),
        ("Manage Beneficiaries", "Add, edit or remove beneficiaries", False),
        ("Mobile Payment", "Wallet transfer within NBO", False),
        ("Scheduled Transfers", "View or cancel pending transfers", False),
        ("Transfer History", "All past online transfers", False),
        ("Remittance Alert", "Set up SMS alert for currency rates", False),
    ]

    for idx, (title, sub, clickable) in enumerate(items):
        col_a, col_b = st.columns([9, 1])
        with col_a:
            st.markdown(f"""
            <div style="background:white;padding:14px 16px;border-bottom:1px solid #F3F4F6;">
              <div style="font-size:15px;font-weight:600;color:#111827;">{title}</div>
              <div style="font-size:13px;color:#6B7280;margin-top:3px;">{sub}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if clickable:
                if st.button("›", key=f"dash_{idx}"):
                    go("beneficiary")
            else:
                st.markdown("<div style='text-align:center;font-size:22px;color:#1456D8;padding-top:14px;'>›</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()

# ════════════════════════════════════════════════════════════════════════════════
# SCREEN: BENEFICIARY
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "beneficiary":
    top_bar(show_user=True)
    section_title("Select Beneficiary")
    st.markdown('<div style="padding:14px 16px;">', unsafe_allow_html=True)
    st.markdown(steps_html(1), unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .bene-card {
        background: white;
        border-radius: 14px;
        padding: 14px 14px;
        border: 1px solid #E5E7EB;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.18s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        min-height: 112px;
    }
    .bene-card:hover {
        border-color: #1456D8;
        box-shadow: 0 6px 18px rgba(20, 86, 216, 0.10);
        transform: translateY(-1px);
    }
    .bene-card-title {
        font-size: 15px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 6px;
    }
    .bene-card-sub {
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 6px;
    }
    .bene-card-bank {
        font-size: 12px;
        color: #9CA3AF;
    }
    .bene-chevron {
        color: #1456D8;
        font-size: 24px;
        font-weight: 700;
        margin-left: 12px;
        flex-shrink: 0;
    }

    /* marker before the overlay button */
    .bene-overlay-marker {
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* pull the Streamlit button upward to sit on top of the card */
    .bene-overlay-marker + div[data-testid="element-container"] {
        margin-top: -112px !important;
        margin-bottom: 10px !important;
        position: relative !important;
        z-index: 10 !important;
    }

    .bene-overlay-marker + div[data-testid="element-container"] .stButton {
        width: 100% !important;
    }

    .bene-overlay-marker + div[data-testid="element-container"] .stButton > button {
        width: 100% !important;
        min-height: 112px !important;
        height: 112px !important;
        background: transparent !important;
        color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 14px !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .bene-overlay-marker + div[data-testid="element-container"] .stButton > button:hover,
    .bene-overlay-marker + div[data-testid="element-container"] .stButton > button:focus,
    .bene-overlay-marker + div[data-testid="element-container"] .stButton > button:active {
        background: rgba(20, 86, 216, 0.02) !important;
        color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    benefs = [
        {"name":"Ashok UAE",      "account":"AE280030010423461132001", "bank":"ABU DHABI COMMERCIAL BANK",  "country":"UAE",    "flag":"🇦🇪"},
        {"name":"Canadarestca",   "account":"124552135622",            "bank":"BANQUE NATIONALE DU CANADA", "country":"Canada", "flag":"🇨🇦"},
        {"name":"Deepak India",   "account":"37445612357",             "bank":"STATE BANK OF INDIA",        "country":"India",  "flag":"🇮🇳"},
        {"name":"Gaurav Khrgr",   "account":"10163005395",             "bank":"IDFC FIRST BANK LTD",        "country":"India",  "flag":"🇮🇳"},
        {"name":"Harshit UAE",    "account":"AE920030012054234910001", "bank":"ABU DHABI COMMERCIAL BANK",  "country":"UAE",    "flag":"🇦🇪"},
    ]

    search = st.text_input(
        "Search",
        placeholder="🔍  Search name, bank or country",
        label_visibility="collapsed"
    )

    if search:
        s = search.lower()
        benefs = [b for b in benefs if s in (b["name"] + b["bank"] + b["country"]).lower()]

    for i, b in enumerate(benefs):
        # visual card
        st.markdown(f"""
        <div class="bene-card">
            <div>
                <div class="bene-card-title">{b['flag']}&nbsp;&nbsp;{b['name']}</div>
                <div class="bene-card-sub">{b['account']}</div>
                <div class="bene-card-bank">{b['bank']} · {b['country']}</div>
            </div>
            <div class="bene-chevron"></div>
        </div>
        """, unsafe_allow_html=True)

        # invisible overlay button
        st.markdown('<div class="bene-overlay-marker"></div>', unsafe_allow_html=True)
        if st.button(" Select", key=f"sel_{i}"):
            td = st.session_state.transfer_data
            td["beneficiary_name"] = b["name"]
            td["beneficiary_account"] = b["account"]
            td["beneficiary_bank"] = b["bank"]
            td["beneficiary_country"] = b["country"]
            st.session_state.selected_beneficiary = b["name"]
            go("details")

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
    if st.button("← Back", key="bene_back"):
        go("dashboard")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()

# ════════════════════════════════════════════════════════════════════════════════
# SCREEN: DETAILS
# ════════════════════════════════════════════════════════════════════════════════
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
      <div style="font-size:13px;color:#1456D8;font-weight:500;margin-top:4px;">Balance: OMR {td['available_balance']:.3f}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#EEF2FA;border-radius:14px;padding:12px 14px;margin-bottom:16px;
                border-left:4px solid #12805C;">
      <div style="font-size:12px;color:#6B7280;margin-bottom:3px;">To Beneficiary</div>
      <div style="font-size:15px;font-weight:600;color:#111827;">{td['beneficiary_name']}</div>
      <div style="font-size:13px;color:#374151;">{td['beneficiary_bank']} · {td['beneficiary_country']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="background:#FFF8E7;border-radius:10px;padding:9px 12px;font-size:13px;color:#7D4E00;margin-bottom:14px;">💱  1 OMR = {td["exchange_rate"]} INR</div>', unsafe_allow_html=True)

    st.markdown('<p style="color:#374151;font-size:14px;font-weight:500;margin-bottom:4px;">Amount (INR)</p>', unsafe_allow_html=True)
    amount = st.number_input("amount_lbl", min_value=0.0, value=float(td["amount_inr"]), step=500.0, label_visibility="collapsed")
    td["amount_inr"] = amount

    omr_equiv = amount / td["exchange_rate"]
    st.markdown(f'<div style="font-size:13px;color:#6B7280;margin-top:-8px;margin-bottom:12px;">≈ OMR {omr_equiv:.3f} will be debited</div>', unsafe_allow_html=True)

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
    st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()

# ════════════════════════════════════════════════════════════════════════════════
# SCREEN: CONFIRM
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "confirm":
    td = st.session_state.transfer_data
    omr_debit = td["amount_inr"] / td["exchange_rate"]
    top_bar(show_user=True)
    section_title("Confirm Transaction")
    st.markdown('<div style="padding:14px 16px;">', unsafe_allow_html=True)
    st.markdown(steps_html(3), unsafe_allow_html=True)
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    blue_band("Transaction Details")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;border-radius:0 0 14px 14px;padding:4px 14px 4px;">
      {row_pair("From", td['from_account'])}
      {row_pair("Account", td['from_account_name'])}
      {row_pair("To", td['beneficiary_account'])}
    </div>""", unsafe_allow_html=True)

    blue_band("Transfer Amount")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;border-radius:0 0 14px 14px;padding:4px 14px 4px;">
      {row_pair("Amount", f"INR {td['amount_inr']:,.2f}")}
      {row_pair("Service Charges", "OMR 0.000")}
      {row_pair("Debit Amount", f"OMR {omr_debit:.3f}")}
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
    st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()

# ════════════════════════════════════════════════════════════════════════════════
# SCREEN: OTP
# ════════════════════════════════════════════════════════════════════════════════
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

    st.markdown('<p style="color:#374151;font-size:14px;font-weight:500;margin-bottom:4px;">Enter 6-digit OTP</p>', unsafe_allow_html=True)
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
            # OLD (remove this):
            st.session_state.risk_result = compute_risk(st.session_state.transfer_data)

            # # NEW (replace with this):
            # with st.spinner("Verifying transaction..."):
            #     st.session_state.risk_result = run_fraud_check(st.session_state.transfer_data)
            go("result")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()

# ════════════════════════════════════════════════════════════════════════════════
# SCREEN: RESULT
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "result":
    td = st.session_state.transfer_data
    result = st.session_state.risk_result or compute_risk(td)
    level = result["level"]

    risk_styles = {
        "Low": ("✅", "#E7F6EF", "#12805C", "#C6EDD9"),
        "Medium": ("⚠️", "#FFF8E7", "#A96900", "#FFE6A0"),
        "High": ("🚨", "#FDECEC", "#B42318", "#FDBDBA"),
    }
    icon, bg, fg, border = risk_styles[level]

    top_bar(show_user=True)
    section_title("Transfer Initiated")
    st.markdown('<div style="padding:16px;">', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#E7F6EF;border-radius:14px;padding:14px;margin-bottom:14px;
                border:1px solid #C6EDD9;text-align:center;">
      <div style="font-size:28px;">✓</div>
      <div style="font-size:16px;font-weight:700;color:#12805C;margin-top:4px;">Transfer Submitted Successfully</div>
      <div style="font-size:13px;color:#374151;margin-top:4px;">INR {td['amount_inr']:,.2f} → {td['beneficiary_name']}</div>
    </div>""", unsafe_allow_html=True)

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

    blue_band("Transfer Summary")
    st.markdown(f"""
    <div style="background:white;border:1px solid #E5E7EB;border-top:none;
                border-radius:0 0 14px 14px;padding:4px 14px 10px;margin-bottom:14px;">
      {row_pair("Beneficiary", td['beneficiary_name'])}
      {row_pair("Amount", f"INR {td['amount_inr']:,.2f}")}
      {row_pair("Country", td['beneficiary_country'])}
      {row_pair("Purpose", td['purpose'])}
      {row_pair("Date", td['transfer_date'])}
    </div>""", unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)
    footer_nav()