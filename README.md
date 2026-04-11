# Fraud Detection Model

This project is a banking fraud-detection demo built around:

- a Streamlit transfer app
- an XGBoost-based fraud predictor
- a YAML rule engine
- a bridge layer that connects the app to the fraud engine
- Gmail alerts for medium/high risk cases
- WhatsApp action-required messages for low risk cases

The main user flow is implemented in [app/app1_updated.py](fraudDetectionModel/app/app1_updated.py), and the fraud orchestration is handled by [integration_layer/fraud_bridge.py](fraudDetectionModel/integration_layer/fraud_bridge.py).

## Project Structure

- [app/app1_updated.py](fraudDetectionModel/app/app1_updated.py)
  Streamlit UI for login, beneficiary selection, transfer details, OTP, and result display.

- [integration_layer/fraud_bridge.py](fraudDetectionModel/integration_layer/fraud_bridge.py)
  Converts app payloads into predictor input, enriches customer and beneficiary details from the Excel dataset, triggers fraud scoring, and sends notifications.

- [core_model/predict_fraud.py](fraudDetectionModel/core_model/predict_fraud.py)
  Real-time scoring engine that combines XGBoost model output with YAML rules.

- [core_model/xGBoost_model_WithFeature.py](fraudDetectionModel/core_model/xGBoost_model_WithFeature.py)
  Training and evaluation pipeline used to generate the saved model artifacts.

- [rule engine/fraud_rules.yaml](fraudDetectionModel/rule%20engine/fraud_rules.yaml)
  Rule-based fraud logic applied alongside the model.

- [communication/gmail_sender.py](fraudDetectionModel/communication/gmail_sender.py)
  Gmail SMTP sender used for email alerts.

- [communication/whatsapp_sender.py](fraudDetectionModel/communication/whatsapp_sender.py)
  Twilio WhatsApp sender used for low risk customer notifications.

- [fraud_detection_synthetic_data.xlsx](fraudDetectionModel/fraud_detection_synthetic_data.xlsx)
  Synthetic source data for customers, beneficiaries, transactions, and login history.

- [output/xgb_fraud_model.json](fraudDetectionModel/output/xgb_fraud_model.json)
  Saved XGBoost model used by the predictor.

- [output/model_metadata.json](fraudDetectionModel/output/model_metadata.json)
  Feature metadata and label encoder information for inference.

## Current Notification Logic

- `Low` risk:
  sends a WhatsApp action-required message to the customer

- `Medium` risk:
  sends an email alert

- `High` risk:
  sends an escalated email alert

The bridge decides this in [integration_layer/fraud_bridge.py](fraudDetectionModel/integration_layer/fraud_bridge.py).

## Input Contract Between App And Bridge

The app now sends transfer data in a format compatible with `run_fraud_check(...)`.

Important fields passed from the app:

- `customer_id`
- `beneficiary_id`
- `amount_usd`
- `channel`
- `device_used`
- `purpose`
- `remark`
- `transfer_type`
- `transfer_date`
- `transaction_timestamp`

The bridge enriches the payload using the Excel-backed model data and fills customer and beneficiary details automatically.

## Requirements

Core dependencies from [requirements.txt](fraudDetectionModel/requirements.txt):

- `numpy>=1.21`
- `pandas>=2.0`
- `scikit-learn>=1.0`
- `xgboost>=1.7`
- `streamlit>=1.35`
- `pyyaml>=6.0`

## Setup

From the project root:

```bash
python -m pip install -r requirements.txt
```

If you use a virtual environment or Conda environment, activate it first.

## Important Run Instruction

Use this command to start the app:

```bash
python -m streamlit run app/app1_updated.py
```

Do not rely on:

```bash
streamlit run app/app1_updated.py
```

unless you are sure it uses the same Python interpreter as your working environment.

This matters because `xgboost` may fail to load in one environment and work in another. Running Streamlit as a Python module ensures it uses the same interpreter as:

```bash
python
```

## Running The App

From the project root:

```bash
python -m streamlit run app/app1_updated.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Running The Bridge Directly

You can also run the bridge self-test directly:

```bash
python integration_layer/fraud_bridge.py
```

This is useful for checking:

- model loading
- payload enrichment
- risk scoring
- email / WhatsApp notification logic

## Running The Predictor Directly

To test the fraud predictor on its own:

```bash
python core_model/predict_fraud.py
```

## Retraining The Model

To rebuild the XGBoost model and metadata:

```bash
python core_model/xGBoost_model_WithFeature.py
```

This updates artifacts in `output/`.

## Gmail Configuration

Email alerts use [communication/gmail_sender.py](fraudDetectionModel/communication/gmail_sender.py).

Current implementation uses configured Gmail SMTP credentials in code. For a safer setup, move these to environment variables.

Typical Gmail requirements:

- enable 2-Step Verification
- generate an App Password
- use that App Password for SMTP login

## WhatsApp Configuration

WhatsApp notifications use [communication/whatsapp_sender.py](fraudDetectionModel/communication/whatsapp_sender.py).

Set these environment variables before running:

```bash
export WHATSAPP_ACCOUNT_SID="your_twilio_account_sid"
export WHATSAPP_AUTH_TOKEN="your_twilio_auth_token"
export WHATSAPP_FROM_NUMBER="+14155238886"
```

Optional:

```bash
export WHATSAPP_DEFAULT_COUNTRY_CODE="+968"
export WHATSAPP_TO_NUMBER="+968XXXXXXXX"
```

`WHATSAPP_TO_NUMBER` is useful for forcing all test messages to a single phone number.

## Debugging

The app, bridge, and predictor currently include debug `print(...)` statements.

When you run:

```bash
python -m streamlit run app/app1_updated.py
```

look at the same terminal for logs from:

- `app1_updated`
- `fraud_bridge`
- `predict_fraud`

Useful examples:

- whether `run_fraud_check` imported successfully
- the transfer payload sent by the app
- the enriched payload used by the bridge
- the model input sent to the predictor
- the final risk decision
- notification success/failure

## Common Issues

### `fraud_bridge` import fails in the app

If you see an import failure mentioning XGBoost or `libxgboost.dylib`, the Streamlit command is probably using a different Python environment.

Use:

```bash
python -m streamlit run app/app1_updated.py
```

### Model unavailable: `DataFrame index must be unique for orient='index'`

This usually means duplicate IDs exist in the Excel data:

- duplicate `customer_id` in the `Customers` sheet
- or duplicate `beneficiary_id` in the `Beneficiaries` sheet

Clean the Excel data so those ID columns are unique.

### Email works directly but not from the app

Check that the app is actually importing and calling `run_fraud_check(...)`. The terminal logs will show this clearly.

## Notes

- The UI now uses USD as the main amount display.
- OMR is shown only as a display equivalent on the details screen.
- Fraud logic uses USD, not OMR.
- Customer and beneficiary details are enriched in the bridge from the Excel dataset.
