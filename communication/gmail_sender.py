"""
gmail_sender.py
---------------
Send email from Python using Gmail SMTP and an App Password.

Setup:
1) Enable 2-Step Verification on your Gmail account
2) Generate a Gmail App Password
3) Update SENDER_EMAIL and APP_PASSWORD variables

Configuration:
    SENDER_EMAIL = "your_email@gmail.com"
    APP_PASSWORD = "your_16_char_app_password"

Run examples:
    python communication/gmail_sender.py

Or import into your project:
    from communication.gmail_sender import send_email
"""

import os
import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional, Union
from urllib.parse import quote

# Email configuration - Update these with your Gmail credentials
SENDER_EMAIL = "mayank.pgcsmp01@iimtrichy.ac.in"  # Replace with your Gmail address
APP_PASSWORD = "psepehbowhdqbtpl"    # Replace with your 16-character Gmail App Password

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def _normalize_recipients(to: Union[str, Iterable[str]]) -> str:
    if isinstance(to, str):
        return to
    return ", ".join([addr.strip() for addr in to if str(addr).strip()])


def send_email(
    subject: str,
    body_text: str,
    to: Union[str, Iterable[str]],
    sender_email: Optional[str] = None,
    app_password: Optional[str] = None,
    cc: Optional[Union[str, Iterable[str]]] = None,
    bcc: Optional[Union[str, Iterable[str]]] = None,
    body_html: Optional[str] = None,
    attachments: Optional[Iterable[Union[str, Path]]] = None,
    smtp_server: str = SMTP_SERVER,
    smtp_port: int = SMTP_PORT,
) -> None:
    sender_email = sender_email or SENDER_EMAIL
    app_password = app_password or APP_PASSWORD

    if not sender_email or sender_email == "your_email@gmail.com":
        raise ValueError("Missing sender email. Update SENDER_EMAIL variable.")
    if not app_password or app_password == "your_app_password":
        raise ValueError("Missing Gmail app password. Update APP_PASSWORD variable.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = _normalize_recipients(to)

    if cc:
        msg["Cc"] = _normalize_recipients(cc)

    all_bcc = _normalize_recipients(bcc) if bcc else None

    msg.set_content(body_text)

    if body_html:
        msg.add_alternative(body_html, subtype="html")

    if attachments:
        for file_path in attachments:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Attachment not found: {path}")

            mime_type, _ = mimetypes.guess_type(str(path))
            if mime_type:
                maintype, subtype = mime_type.split("/", 1)
            else:
                maintype, subtype = "application", "octet-stream"

            with path.open("rb") as f:
                file_data = f.read()

            msg.add_attachment(
                file_data,
                maintype=maintype,
                subtype=subtype,
                filename=path.name,
            )

    recipients = [msg["To"]]
    if cc:
        recipients.append(msg["Cc"])
    if all_bcc:
        recipients.append(all_bcc)

    recipient_list = []
    for item in recipients:
        if item:
            recipient_list.extend([x.strip() for x in item.split(",") if x.strip()])

    with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) if smtp_port == 465 else smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        if smtp_port != 465:
            server.starttls()
        server.login(sender_email, app_password)
        server.send_message(msg, from_addr=sender_email, to_addrs=recipient_list)


def send_fraud_alert(
    recipient_email: str,
    customer_name: str,
    account_last_4: str,
    amount: float,
    merchant_name: str,
    transaction_date: str,
    currency: str = "OMR",
    bank_name: str = "National bank of oman",
    support_phone: str = "+968 2477 7777",
) -> None:
    """
    Sends a customer-facing fraud alert email with professional formatting.
    """
    
    # 1. Professional Subject Line
    subject = f"Action Required: Suspicious Transaction on Account ending in {account_last_4}"
    
    # 2. Plain Text Fallback (for email clients that block HTML)
    body_text = f"""Dear {customer_name},

We detected unusual activity on your {bank_name} account ending in {account_last_4}. For your protection, we want to verify a recent transaction.

Transaction Details:
- Date: {transaction_date}
- Merchant: {merchant_name}
- Amount: {amount:.2f} {currency}

Was this you?
If you recognize this transaction, no further action is needed. 
If you DO NOT recognize this transaction, please call our Fraud Prevention Department immediately at {support_phone}.

Sincerely,
{bank_name} Fraud Prevention Team
"""

    # 3. Clean, Modern HTML Email Body
    body_html = f"""
    <!DOCTYPE html>
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; margin: 0; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
          
          <div style="background-color: #234DC6; color: #ffffff; padding: 20px; text-align: center;">
            <h2 style="margin: 0;">{bank_name}</h2>
            <p style="margin: 5px 0 0 0; font-size: 14px;">Important Security Alert</p>
          </div>
          
          <div style="padding: 30px;">
            <p style="font-size: 16px; color: #333333;">Dear <b>{customer_name}</b>,</p>
            <p style="font-size: 16px; color: #555555; line-height: 1.5;">
              We detected some unusual activity on your account ending in <b>{account_last_4}</b>. For your protection, we want to verify the following recent transaction:
            </p>
            
            <div style="background-color: #f9f9f9; border-left: 4px solid #234DC6; padding: 15px; margin: 25px 0;">
              <table style="width: 100%; border-collapse: collapse;">
                <tr>
                  <td style="padding: 8px 0; color: #777777; width: 40%;"><b>Date:</b></td>
                  <td style="padding: 8px 0; color: #333333;">{transaction_date}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; color: #777777;"><b>Merchant:</b></td>
                  <td style="padding: 8px 0; color: #333333;">{merchant_name}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; color: #777777;"><b>Amount:</b></td>
                  <td style="padding: 8px 0; color: #234DC6; font-size: 18px;"><b>{amount:.2f} {currency}</b></td>
                </tr>
              </table>
            </div>
            
            <h3 style="color: #333333;">Was this you?</h3>
            <p style="font-size: 16px; color: #555555; line-height: 1.5;">
              If you <b>recognize</b> this transaction, no further action is required and your account is secure.
            </p>
            <p style="font-size: 16px; color: #555555; line-height: 1.5;">
              If you <b>DO NOT recognize</b> this transaction, please contact us immediately to secure your account.
            </p>
            
            <div style="text-align: center; margin: 30px 0;">
              <a href="tel:{support_phone}" style="background-color: #234DC6; color: #ffffff; text-decoration: none; padding: 12px 25px; border-radius: 4px; font-size: 16px; font-weight: bold; display: inline-block;">Call {support_phone}</a>
            </div>
          </div>
          
          <div style="background-color: #eeeeee; padding: 20px; text-align: center; font-size: 12px; color: #888888;">
            <p style="margin: 0;">Please do not reply to this email. This inbox is not monitored.</p>
            <p style="margin: 5px 0 0 0;">&copy; 2026 {bank_name}. All rights reserved.</p>
          </div>
          
        </div>
      </body>
    </html>
    """

    # 4. Trigger the email
    send_email(
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        to=recipient_email,
    )


def send_high_risk_confirmation_email(
    recipient_email: str,
    customer_name: str,
    account_last_4: str,
    amount: float,
    beneficiary_name: str,
    transaction_date: str,
    transaction_reference: str,
    currency: str = "USD",
    bank_name: str = "National Bank of Oman",
    reply_to_email: Optional[str] = "nbo.bank@om.nbo.om",
    beneficiary_country: str = "",
    purpose: str = "",
    channel: str = "",
    support_email: Optional[str] = None,
    support_phone: str = "+968 2477 7777",
) -> None:
    """Send a red-header confirmation email for high-risk transactions."""
    reply_to_email = reply_to_email or SENDER_EMAIL
    support_email = support_email or reply_to_email
    subject = f"Action Required: Confirm Transaction Ref {transaction_reference}"
    confirm_subject = quote(f"Customer confirmation for transaction {transaction_reference}")
    confirm_body_lines = [
        f"To: {bank_name} Fraud Review Team,",
        "",
        "I confirm that I authorize the transaction referenced below and request the bank to proceed with processing.",
        "",
        f"Transaction Reference Number: {transaction_reference}",
        f"Transaction Date: {transaction_date}",
        f"Debit Account: Account ending in {account_last_4}",
        f"Beneficiary: {beneficiary_name}",
        f"Amount: {amount:.2f} {currency}",
    ]
    if beneficiary_country:
        confirm_body_lines.append(f"Beneficiary Country: {beneficiary_country}")
    if purpose:
        confirm_body_lines.append(f"Purpose: {purpose}")
    if channel:
        confirm_body_lines.append(f"Channel: {channel}")
    confirm_body_lines.extend([
        "",
        "Confirmation Statement:",
        "I confirm the transaction.",
        "",
        f"Customer Name: {customer_name}",
    ])
    confirm_body = quote("\n".join(confirm_body_lines))
    mailto_href = f"mailto:{reply_to_email}?subject={confirm_subject}&body={confirm_body}"
    country_line = f"- Beneficiary Country: {beneficiary_country}\n" if beneficiary_country else ""
    purpose_line = f"- Purpose: {purpose}\n" if purpose else ""
    channel_line = f"- Channel: {channel}\n" if channel else ""
    country_html = (
        f"""
                <tr>
                  <td style="padding: 8px 0; color: #6b7280;"><strong>Beneficiary Country</strong></td>
                  <td style="padding: 8px 0; color: #111827;">{beneficiary_country}</td>
                </tr>
        """
        if beneficiary_country else ""
    )
    purpose_html = (
        f"""
                <tr>
                  <td style="padding: 8px 0; color: #6b7280;"><strong>Purpose</strong></td>
                  <td style="padding: 8px 0; color: #111827;">{purpose}</td>
                </tr>
        """
        if purpose else ""
    )
    channel_html = (
        f"""
                <tr>
                  <td style="padding: 8px 0; color: #6b7280;"><strong>Channel</strong></td>
                  <td style="padding: 8px 0; color: #111827;">{channel}</td>
                </tr>
        """
        if channel else ""
    )

    body_text = f"""Dear {customer_name},

We have temporarily placed a high-risk transaction on hold for additional verification on your {bank_name} account ending in {account_last_4}.

Transaction Details:
- Reference Number: {transaction_reference}
- Date: {transaction_date}
- Beneficiary: {beneficiary_name}
- Account ending: {account_last_4}
- Amount: {amount:.2f} {currency}
{country_line}{purpose_line}{channel_line}
To authorise processing of this transaction, please confirm it by replying with the exact text below:
I confirm the transaction.
Transaction Reference Number: {transaction_reference}

If you do not recognise this transaction, do not confirm it. Please contact {bank_name} immediately at {support_phone} or {support_email}.
"""

    body_html = f"""
    <!DOCTYPE html>
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f6fb; margin: 0; padding: 24px;">
        <div style="max-width: 680px; margin: 0 auto; background-color: #ffffff; border-radius: 14px; overflow: hidden; box-shadow: 0 12px 36px rgba(15, 23, 42, 0.12); border: 1px solid #e5e7eb;">
          <div style="background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%); color: #ffffff; padding: 24px 28px;">
            <div style="font-size: 12px; letter-spacing: 1.2px; font-weight: 700; opacity: 0.92; text-transform: uppercase;">High Risk Transaction Review</div>
            <h1 style="margin: 10px 0 6px 0; font-size: 28px; line-height: 1.2;">Customer Confirmation Required</h1>
            <p style="margin: 0; font-size: 15px; line-height: 1.6; opacity: 0.95;">
              The transaction below is currently on hold and will be processed only after we receive your confirmation.
            </p>
          </div>

          <div style="padding: 30px 28px;">
            <p style="font-size: 16px; color: #1f2937; margin-top: 0;">Dear <strong>{customer_name}</strong>,</p>
            <p style="font-size: 16px; color: #4b5563; line-height: 1.7; margin-bottom: 18px;">
              As part of our fraud prevention controls, we identified elevated risk indicators linked to a pending transaction on your account ending in <strong>{account_last_4}</strong>.
              To protect your account, we require direct confirmation from you before the transaction can proceed.
            </p>

            <div style="background-color: #fff7f7; border: 1px solid #fecaca; border-radius: 12px; padding: 18px 20px; margin: 24px 0 18px 0;">
              <div style="font-size: 13px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; color: #b91c1c; margin-bottom: 12px;">
                Transaction Summary
              </div>
              <table style="width: 100%; border-collapse: collapse;">
                <tr>
                  <td style="padding: 8px 0; color: #6b7280; width: 42%;"><strong>Transaction Reference</strong></td>
                  <td style="padding: 8px 0; color: #111827; font-weight: 700;">{transaction_reference}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; color: #6b7280;"><strong>Transaction Date</strong></td>
                  <td style="padding: 8px 0; color: #111827;">{transaction_date}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; color: #6b7280;"><strong>Debit Account</strong></td>
                  <td style="padding: 8px 0; color: #111827;">Account ending in {account_last_4}</td>
                </tr>
                <tr>
                  <td style="padding: 8px 0; color: #6b7280;"><strong>Beneficiary</strong></td>
                  <td style="padding: 8px 0; color: #111827;">{beneficiary_name}</td>
                </tr>
                {country_html}
                {purpose_html}
                {channel_html}
                <tr>
                  <td style="padding: 8px 0; color: #6b7280;"><strong>Amount</strong></td>
                  <td style="padding: 8px 0; color: #b91c1c; font-size: 18px;"><strong>{amount:.2f} {currency}</strong></td>
                </tr>
              </table>
            </div>

            <div style="background-color: #fff1f2; border: 1px solid #fecdd3; border-radius: 12px; padding: 16px 18px; margin: 0 0 22px 0;">
              <div style="font-size: 15px; font-weight: 700; color: #9f1239; margin-bottom: 8px;">Required Action</div>
              <div style="font-size: 15px; color: #4b5563; line-height: 1.7;">
                If you authorised this transaction, please use the confirmation button below. A new email draft will open with the required confirmation statement and transaction reference number already completed for you.
              </div>
            </div>

            <div style="text-align: center; margin: 28px 0 22px;">
              <a href="{mailto_href}" style="background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%); color: #ffffff; text-decoration: none; padding: 15px 28px; border-radius: 999px; font-size: 16px; font-weight: 700; display: inline-block; box-shadow: 0 10px 24px rgba(220, 38, 38, 0.25);">Confirm Transaction by Email</a>
            </div>

            <div style="font-size: 14px; color: #6b7280; line-height: 1.7; margin-top: 4px;">
              <strong style="color: #374151;">If you do not recognise this transaction:</strong>
              do not confirm it. Please contact our fraud support team immediately at
              <a href="tel:{support_phone}" style="color: #b91c1c; text-decoration: none; font-weight: 700;">{support_phone}</a>
              or email
              <a href="mailto:{support_email}" style="color: #b91c1c; text-decoration: none; font-weight: 700;">{support_email}</a>.
            </div>
          </div>

          <div style="background-color: #f8fafc; border-top: 1px solid #e5e7eb; padding: 18px 28px;">
            <div style="font-size: 12px; color: #6b7280; line-height: 1.7; text-align: center;">
              This message was generated as part of {bank_name}'s fraud prevention workflow.<br>
              Please retain the transaction reference for any future correspondence.
            </div>
          </div>
        </div>
      </body>
    </html>
    """

    send_email(
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        to=recipient_email,
    )


if __name__ == "__main__":
    try:
        # Replaced internal metrics (mobile_number, risk_level) with the new customer parameters
        send_fraud_alert(
           recipient_email="nbo.bank@gmail.com",
           customer_name="Valued Customer", 
           account_last_4="8946", 
           amount=100.0,
           merchant_name="Unknown Online Merchant",
           transaction_date="April 10, 2026 02:00 AM",
        )    
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")
        print("Make sure to update SENDER_EMAIL and APP_PASSWORD variables at the top of the file.")
