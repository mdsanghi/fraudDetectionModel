import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(page_title="NBO Transfer POC", layout="centered")

# ---------- Session State ----------
def init_state():
    defaults = {
        "screen": "login",
        "passcode": "",
        "selected_beneficiary": None,
        "otp": "",
        "transfer_data": {
            "customer_name": "MAYANK MANOJ SANGHI",
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

# ---------- Theme / CSS ----------
PRIMARY = "#1456D8"
LIGHT_BG = "#EEF2FA"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "#6B7280"
BORDER = "#D7DCE5"
SUCCESS = "#12805C"
WARN = "#A96900"
DANGER = "#B42318"

st.markdown(f"""
<style>
    .stApp {{
        background: {LIGHT_BG};
        color: {TEXT};
    }}
    .block-container {{
        max-width: 430px;
        padding-top: 0rem;
        padding-bottom: 1rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }}
    .mobile-shell {{
        background: {LIGHT_BG};
        min-height: 100vh;
        border-radius: 26px;
        overflow: hidden;
        box-shadow: 0 8px 30px rgba(0,0,0,.10);
    }}
    .top-blue {{
        background: {PRIMARY};
        color: white;
        padding: 14px 22px 18px 22px;
    }}
    .status-bar {{
        display:flex;justify-content:space-between;align-items:center;
        font-size: 14px;font-weight:700;margin-bottom: 10px;
    }}
    .brand {{
        text-align:center;font-size:48px;font-weight:800;line-height:1;
        letter-spacing:.5px;
    }}
    .brand-sub {{
        text-align:center;font-size:18px;font-weight:700;opacity:.95;margin-bottom:2px;
    }}
    .user-row {{
        display:flex;justify-content:space-between;align-items:center;
        font-size:18px;font-weight:500;padding-top:12px;
    }}
    .section-title {{
        color: {PRIMARY};font-size:17px;font-weight:500;padding:14px 22px;background:#DDE4F2;
        border-bottom:1px solid {BORDER};
    }}
    .content {{ padding: 16px 18px 26px 18px; }}
    .menu-item {{
        background: {CARD};border-top:1px solid {BORDER};padding:20px 16px;
        font-size:18px;font-weight:500;
    }}
    .menu-sub {{ color:#444; font-size:15px; margin-top:6px; line-height:1.2; }}
    .step-wrap {{ text-align:center; padding: 8px 0 14px 0; }}
    .step {{
        display:inline-flex; align-items:center; justify-content:center;
        width:42px; height:42px; border-radius:50%; margin:0 5px;
        font-weight:700; font-size:22px; background:#D9D9D9; color:white;
    }}
    .step.active {{ background:{PRIMARY}; }}
    .card {{
        background:{CARD}; border:1px solid {BORDER}; border-radius:18px;
        padding: 14px; margin-bottom: 14px;
    }}
    .label-sm {{ color:#444; font-size:14px; }}
    .value-lg {{ color:{TEXT}; font-size:18px; font-weight:500; }}
    .blue-band {{ background:{PRIMARY}; color:white; padding:8px 12px; font-size:18px; font-weight:600; }}
    .summary-row {{ display:flex; justify-content:space-between; gap:16px; padding:10px 0; border-bottom:1px solid #eaeaea; }}
    .summary-row:last-child {{ border-bottom:none; }}
    .summary-key {{ width:34%; font-size:16px; }}
    .summary-val {{ width:66%; font-size:16px; font-weight:500; }}
    .otp-box {{
        text-align:center; background:white; border:1px solid {BORDER}; border-radius:18px; padding:24px;
    }}
    .risk-low, .risk-medium, .risk-high {{ padding:14px 16px; border-radius:14px; font-weight:700; margin-top:10px; }}
    .risk-low {{ background:#E7F6EF; color:{SUCCESS}; }}
    .risk-medium {{ background:#FFF2D8; color:{WARN}; }}
    .risk-high {{ background:#FDECEC; color:{DANGER}; }}
    .footer-nav {{
        position: sticky; bottom: 0; background: {PRIMARY}; color: white; padding: 14px 10px;
        display:flex; justify-content:space-around; font-size:14px;
    }}
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextArea"] textarea {{
        border-radius: 14px !important; border:1px solid {BORDER} !important; padding: 10px 12px !important;
        font-size: 17px !important;
    }}
    div[data-baseweb="select"] > div {{
        border-radius: 14px !important; border:1px solid {BORDER} !important;
        min-height: 52px;
    }}
    .stButton > button {{
        width:100%; border-radius: 28px; min-height: 52px; font-size: 18px; font-weight: 600;
        border: none; background: {PRIMARY}; color: white;
    }}
    .outline-btn > button {{
        background: white !important; color: {PRIMARY} !important; border: 2px solid {PRIMARY} !important;
    }}
    .tiny-note {{ color: {MUTED}; font-size: 13px; }}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def shell_start(show_footer=False):
    st.markdown('<div class="mobile-shell">', unsafe_allow_html=True)
    if show_footer:
        footer()

def shell_end():
    st.markdown('</div>', unsafe_allow_html=True)


def top_header(user_name=True):
    st.markdown(f'''
    <div class="top-blue">
      <div class="status-bar"><span>8:56</span><span>📶 Wi‑Fi 🔋88</span></div>
      <div class="brand-sub">البنك الوطني العماني</div>
      <div class="brand">NBO</div>
      {f'<div class="user-row"><span>{st.session_state.transfer_data["customer_name"]}</span><span>Rewards</span></div>' if user_name else ''}
    </div>
    ''', unsafe_allow_html=True)


def footer():
    st.markdown('''
    <div class="footer-nav">
      <div>Videos</div>
      <div>Offers</div>
      <div>Scan&Pay</div>
      <div>Notifications</div>
      <div>More</div>
    </div>
    ''', unsafe_allow_html=True)


def steps(active):
    cls = lambda i: "step active" if i == active else "step"
    st.markdown(f'''
    <div class="step-wrap">
      <span class="{cls(1)}">1</span>
      <span class="{cls(2)}">2</span>
      <span class="{cls(3)}">3</span>
    </div>
    ''', unsafe_allow_html=True)


def go(screen):
    st.session_state.screen = screen
    st.rerun()


def compute_risk(data):
    score = 0
    triggered = []
    amount_inr = float(data.get("amount_inr", 0))
    beneficiary_country = data.get("beneficiary_country", "")
    purpose = data.get("purpose", "")

    if amount_inr >= 20000:
        score += 30
        triggered.append("High transfer amount")
    elif amount_inr >= 10000:
        score += 15
        triggered.append("Moderately high amount")

    if beneficiary_country.lower() not in ["oman", "india"]:
        score += 35
        triggered.append("Less common beneficiary country")
    elif beneficiary_country.lower() == "india":
        score += 10
        triggered.append("Cross-border transfer")

    if purpose.lower() in ["gift", "personal", "others"]:
        score += 10
        triggered.append("Generic transfer purpose")

    if len(str(data.get("remark", "")).strip()) == 0:
        score += 5
        triggered.append("No remark provided")

    if score < 20:
        level = "Low"
    elif score < 45:
        level = "Medium"
    else:
        level = "High"
    return {"score": score, "level": level, "triggered": triggered}

# ---------- Screens ----------
if st.session_state.screen == "login":
    shell_start()
    st.markdown(f'''
    <div style="background:{PRIMARY}; color:white; min-height:100vh; padding:18px 22px 24px 22px;">
      <div class="status-bar"><span>8:55</span><span>📶 Wi‑Fi 🔋88</span></div>
      <div style="padding-top:24px;"></div>
      <div class="brand-sub">البنك الوطني العماني</div>
      <div class="brand">NBO</div>
      <div style="font-size:24px; margin-top:70px; margin-bottom:22px;">Enter your passcode to login</div>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns([1,1])
    with c1:
        passcode = st.text_input("Passcode", type="password", value=st.session_state.passcode, placeholder="Enter 4-digit passcode")
        st.session_state.passcode = passcode
    with c2:
        st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
        if st.button("Login / Face ID"):
            go("dashboard")

    st.markdown('''
      <div style="text-align:center; margin-top:24px; font-size:16px;">New User / Forgot Passcode</div>
      <div style="text-align:center; margin-top:26px; font-size:16px;">- OR -</div>
      <div style="text-align:center; margin-top:20px; font-size:34px;">🙂</div>
      <div style="text-align:center; font-size:18px;">Face ID</div>
      <div style="height:260px;"></div>
    </div>
    ''', unsafe_allow_html=True)
    footer()
    shell_end()

elif st.session_state.screen == "dashboard":
    shell_start()
    top_header(True)
    st.markdown('<div class="section-title">Fund Transfer</div>', unsafe_allow_html=True)
    items = [
        ("Within My Accounts", "Between any of your NBO Accounts"),
        ("Instant Transfer", "India, UAE, Pakistan, Bangladesh, Sri Lanka, Egypt, Philippines, Indonesia"),
        ("To Other Accounts", "To Accounts within NBO, Local Banks or other Banks worldwide"),
        ("Manage Beneficiaries", "Manage Beneficiaries"),
        ("Mobile Payment", "to wallet, within NBO wallet"),
        ("Manage Scheduled Transfers", "View/Cancel Pending Transfers"),
        ("Transfer History", "View all past online transfers"),
        ("Remittance Alert", "Set up SMS alert for currency rate"),
    ]
    st.markdown('<div class="content">', unsafe_allow_html=True)
    for idx, (title, sub) in enumerate(items):
        cols = st.columns([8,1])
        with cols[0]:
            st.markdown(f'<div class="menu-item">{title}<div class="menu-sub">{sub}</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            if title == "Instant Transfer":
                if st.button("›", key=f"go_{idx}"):
                    go("beneficiary")
            else:
                st.markdown("<div style='font-size:32px; color:#1456D8; text-align:center;'>›</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    shell_end()

elif st.session_state.screen == "beneficiary":
    shell_start()
    top_header(True)
    st.markdown('<div class="section-title">Select Beneficiary</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)
    steps(1)
    search = st.text_input("Search by keyword", placeholder="Search beneficiary or country")
    benefs = [
        {"name":"ashok UAE", "account":"AE280030010423461132001", "bank":"ABU DHABI COMMERCIAL BANK", "country":"UAE"},
        {"name":"canadarestca", "account":"124552135622", "bank":"BANQUE NATIONALE DU CANADA", "country":"Canada"},
        {"name":"deepak india", "account":"37445612357", "bank":"STATE BANK OF INDIA", "country":"India"},
        {"name":"gaurav khrgr", "account":"10163005395", "bank":"IDFC FIRST BANK LTD", "country":"India"},
        {"name":"harshit UAE", "account":"AE920030012054234910001", "bank":"ABU DHABI COMMERCIAL BANK", "country":"UAE"},
    ]
    if search:
        benefs = [b for b in benefs if search.lower() in (b['name'] + b['country'] + b['bank']).lower()]
    for i, b in enumerate(benefs):
        st.markdown(f'<div class="card"><div class="value-lg">{b["name"]}</div><div class="label-sm">{b["account"]}<br>{b["bank"]} · {b["country"]}</div></div>', unsafe_allow_html=True)
        if st.button(f"Select {b['name']}", key=f"sel_{i}"):
            st.session_state.selected_beneficiary = b
            td = st.session_state.transfer_data
            td["beneficiary_name"] = b["name"]
            td["beneficiary_account"] = b["account"]
            td["beneficiary_bank"] = b["bank"]
            td["beneficiary_country"] = b["country"]
            go("details")
    st.markdown('</div>', unsafe_allow_html=True)
    shell_end()

elif st.session_state.screen == "details":
    td = st.session_state.transfer_data
    shell_start()
    top_header(True)
    st.markdown('<div class="section-title">Transfer To Other Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)
    steps(2)

    st.markdown('<div class="label-sm">From</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="value-lg">{td["from_account"]}<br>{td["from_account_name"]}</div><div class="label-sm">Available Balance: OMR {td["available_balance"]:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="label-sm">To</div>', unsafe_allow_html=True)
    td["beneficiary_name"] = st.text_input("Beneficiary", value=td["beneficiary_name"], label_visibility="collapsed")

    st.markdown(f'<div class="tiny-note">Exchange Rate: 1 OMR = {td["exchange_rate"]} INR</div>', unsafe_allow_html=True)
    amount = st.number_input("Enter Amount to be Paid (INR)", min_value=0.0, value=float(td["amount_inr"]), step=100.0)
    td["amount_inr"] = amount

    transfer_type = st.radio("When do you want to transfer", ["Transfer Immediately", "Select Future date"], index=0 if td["transfer_type"] == "Transfer Immediately" else 1)
    td["transfer_type"] = transfer_type
    if transfer_type == "Select Future date":
        td["transfer_date"] = str(st.date_input("Transfer date", value=date.today()))
    else:
        td["transfer_date"] = str(date.today())

    td["remark"] = st.text_input("Remark", value=td["remark"], placeholder="(Optional) Enter Remark")
    td["beneficiary_town"] = st.text_input("Beneficiary's Town", value=td["beneficiary_town"])
    purposes = ["Family maintenance", "Education", "Medical", "Gift", "Personal", "Others"]
    td["purpose"] = st.selectbox("Purpose of remittance", purposes, index=purposes.index(td["purpose"]) if td["purpose"] in purposes else 0)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go("beneficiary")
    with c2:
        if st.button("Next"):
            go("confirm")
    st.markdown('</div>', unsafe_allow_html=True)
    shell_end()

elif st.session_state.screen == "confirm":
    td = st.session_state.transfer_data
    shell_start()
    top_header(True)
    st.markdown('<div class="section-title">CONFIRM YOUR TRANSACTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)
    steps(3)

    omr_debit = td["amount_inr"] / td["exchange_rate"]

    st.markdown('<div class="blue-band">Transaction Details</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="card" style="border-radius:0 0 18px 18px; border-top:none; margin-top:0;">
      <div class="summary-row"><div class="summary-key">From</div><div class="summary-val">{td['from_account']}<br>{td['from_account_name']}</div></div>
      <div class="summary-row"><div class="summary-key">To</div><div class="summary-val">{td['beneficiary_account']}</div></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="blue-band">Transfer Amount</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="card" style="border-radius:0 0 18px 18px; border-top:none; margin-top:0;">
      <div class="summary-row"><div class="summary-key">Amount</div><div class="summary-val">INR {td['amount_inr']:.2f}</div></div>
      <div class="summary-row"><div class="summary-key">Service Charges</div><div class="summary-val">OMR 0.000</div></div>
      <div class="summary-row"><div class="summary-key">Debit amount</div><div class="summary-val">OMR {omr_debit:.3f}</div></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="blue-band">Beneficiary Details</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="card" style="border-radius:0 0 18px 18px; border-top:none; margin-top:0;">
      <div class="summary-row"><div class="summary-key">Name</div><div class="summary-val">{td['beneficiary_name']}</div></div>
      <div class="summary-row"><div class="summary-key">Account Number</div><div class="summary-val">{td['beneficiary_account']}</div></div>
      <div class="summary-row"><div class="summary-key">Bank and Branch</div><div class="summary-val">{td['beneficiary_bank']}<br>{td['beneficiary_branch']}</div></div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="blue-band">Date and Frequency</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="card" style="border-radius:0 0 18px 18px; border-top:none; margin-top:0;">
      <div class="summary-row"><div class="summary-key">Transfer Type</div><div class="summary-val">{td['transfer_type']}</div></div>
      <div class="summary-row"><div class="summary-key">Date</div><div class="summary-val">{td['transfer_date']}</div></div>
    </div>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go("details")
    with c2:
        if st.button("Confirm"):
            go("otp")
    st.markdown('</div>', unsafe_allow_html=True)
    shell_end()

elif st.session_state.screen == "otp":
    shell_start()
    top_header(True)
    st.markdown('<div class="section-title">OTP Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)
    st.markdown('<div class="otp-box"><div style="font-size:22px; font-weight:700; margin-bottom:8px;">Enter OTP</div><div class="tiny-note">Use any dummy 6-digit OTP for this POC</div></div>', unsafe_allow_html=True)
    otp = st.text_input("OTP", type="password", max_chars=6, placeholder="Enter 6-digit OTP")
    st.session_state.otp = otp
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back"):
            go("confirm")
    with c2:
        if st.button("Submit OTP"):
            st.session_state.risk_result = compute_risk(st.session_state.transfer_data)
            go("result")
    st.markdown('</div>', unsafe_allow_html=True)
    shell_end()

elif st.session_state.screen == "result":
    td = st.session_state.transfer_data
    result = st.session_state.risk_result or compute_risk(td)
    shell_start()
    top_header(True)
    st.markdown('<div class="section-title">Transfer Initiated</div>', unsafe_allow_html=True)
    st.markdown('<div class="content">', unsafe_allow_html=True)

    st.success("Transfer request submitted successfully.")

    risk_class = result['level'].lower()
    css_class = f"risk-{risk_class}"
    st.markdown(f'<div class="{css_class}">{result["level"]} Suspicion · Risk Score {result["score"]}</div>', unsafe_allow_html=True)
    if result['triggered']:
        st.markdown("**Triggered signals**")
        for item in result['triggered']:
            st.write(f"• {item}")

    st.markdown(f'''
    <div class="card">
      <div class="summary-row"><div class="summary-key">Beneficiary</div><div class="summary-val">{td['beneficiary_name']}</div></div>
      <div class="summary-row"><div class="summary-key">Amount</div><div class="summary-val">INR {td['amount_inr']:.2f}</div></div>
      <div class="summary-row"><div class="summary-key">Purpose</div><div class="summary-val">{td['purpose']}</div></div>
      <div class="summary-row"><div class="summary-key">OTP</div><div class="summary-val">******</div></div>
    </div>
    ''', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Make Another Transfer"):
            st.session_state.risk_result = None
            go("dashboard")
    with c2:
        if st.button("Restart Demo"):
            for key in ["passcode", "selected_beneficiary", "otp", "risk_result"]:
                st.session_state.pop(key, None)
            st.session_state.screen = "login"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    shell_end()
