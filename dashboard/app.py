import os
import time
import requests
import pandas as pd
import streamlit as st
import altair as alt

API_URL = os.getenv("FRAUDGUARD_API_URL", "http://127.0.0.1:8000").rstrip("/")
HEALTH_URL = f"{API_URL}/health"
EVALUATE_URL = f"{API_URL}/v1/evaluate-hybrid"
MAX_INR = 25691.16 * 106.06
FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]

st.set_page_config(
    page_title="FraudGuard AI | Transaction Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp { background: #0b1220; color: #edf3ff; font-family: 'DM Sans', sans-serif; }
[data-testid="stHeader"] { background: #0b1220; }
[data-testid="stSidebar"] { background: #111c2e; border-right: 1px solid #263750; }
.block-container { max-width: 1220px; padding-top: 2rem; padding-bottom: 3rem; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f3f7ff !important; }
p, label, .stMarkdown { color: #b5c3d8; }
.hero { padding: 1.35rem 1.5rem; border: 1px solid #2b4160; border-radius: 18px; background: linear-gradient(120deg,#142947,#17203b); margin-bottom: 1.2rem; }
.eyebrow { color: #8fc4ff; font-size: .72rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
.hero h1 { margin: .25rem 0 .35rem; font-size: 2.15rem; }
.hero p { margin: 0; color: #c0cee2; }
.panel { border: 1px solid #2a3d59; border-radius: 15px; background: #111d30; padding: 1rem 1.1rem; }
.result { border: 1px solid #365781; border-radius: 16px; background: linear-gradient(140deg,#172b49,#171f39); padding: 1.25rem; }
.small-note { font-size: .85rem; color: #9eafc7; }
div[data-testid="stButton"] > button[kind="primary"] { background: linear-gradient(90deg,#2877e8,#7659e8); border: 0; border-radius: 10px; font-weight: 700; min-height: 2.8rem; }
div[data-testid="stMetric"] { background: #111d30; border: 1px solid #2a3d59; border-radius: 12px; padding: .8rem; }
[data-testid="stDataFrame"] { border: 1px solid #2a3d59; border-radius: 10px; }
hr { border-color: #2a3d59; }
</style>
""", unsafe_allow_html=True)

if "fg_log" not in st.session_state:
    st.session_state.fg_log = []
if "fg_last_result" not in st.session_state:
    st.session_state.fg_last_result = None

# Check backend/model availability.
health = None
health_error = None
try:
    health_response = requests.get(HEALTH_URL, timeout=5)
    health_response.raise_for_status()
    health = health_response.json()
except requests.RequestException as exc:
    health_error = str(exc)

components = (health or {}).get("components", {})
api_ready = bool(health and health.get("status") == "ready")
hybrid_ready = api_ready and components.get("hybrid_gnn") == "loaded"

with st.sidebar:
    st.markdown("## 🛡️ FraudGuard AI")
    st.caption("Simple transaction risk checker")
    st.divider()
    st.markdown("### System status")
    if hybrid_ready:
        st.success("Hybrid model is ready")
    else:
        st.error("Backend/model not ready")
        if health_error:
            st.caption(health_error)
        elif health:
            st.json(health)
    st.caption(f"API: {API_URL}")
    st.divider()
    st.markdown("### What the result means")
    st.markdown("**APPROVE** — prototype score is below the review threshold.")
    st.markdown("**OTP VERIFY** — prototype score reached the review threshold; an additional check is suggested.")
    st.caption("This prototype does not issue a BLOCK decision or send an actual OTP.")
    st.divider()
    if st.button("Clear this session's history", use_container_width=True):
        st.session_state.fg_log = []
        st.session_state.fg_last_result = None
        st.rerun()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Transaction screening · Research prototype</div>
  <h1>Check a transaction</h1>
  <p>Enter a card reference and amount, run the model, then read the decision and explanation below.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 1. Enter transaction details")
st.info(
    "For this demo, enter a card reference and amount in INR. The model also requires 28 anonymised "
    "features (V1–V28). Quick demo mode fills these with preset sample values; it does not read real bank data."
)

with st.form("transaction_form"):
    c1, c2 = st.columns(2)
    with c1:
        card_id = st.text_input("Card reference", value="CARD_DEMO_001", help="A demo identifier only. Do not enter a real card number.")
    with c2:
        amount = st.number_input(
            "Transaction amount (₹)",
            min_value=0.0,
            max_value=float(MAX_INR),
            value=5000.0,
            step=100.0,
            format="%.2f",
            help="The API converts INR using a fixed project assumption, not a live exchange rate.",
        )

    input_mode = st.radio(
        "How would you like to provide model features?",
        ["Quick demo", "Enter anonymised features (advanced)"],
        horizontal=True,
    )
    features = {name: 0.0 for name in FEATURE_NAMES}
    if input_mode == "Quick demo":
        demo_scenario = st.selectbox(
            "Demo input",
            ["Typical preset", "Unusual-feature preset"],
            help="These are illustrative feature presets, not real transaction records or guaranteed outcomes.",
        )
        if demo_scenario == "Unusual-feature preset":
            features["V1"] = 4.5
            features["V2"] = -3.0
        st.caption("Quick demo fills V1–V28 automatically. Typical preset uses zeros; unusual preset changes V1 and V2.")
    else:
        st.warning("Only use V1–V28 values produced by the same anonymisation/preprocessing pipeline as the model's training data. Merchant, location, PIN, CVV, and ordinary transaction details cannot be converted into these values by this screen.")
        with st.expander("Enter V1–V28 values"):
            feature_cols = st.columns(4)
            for index, name in enumerate(FEATURE_NAMES):
                with feature_cols[index % 4]:
                    features[name] = st.number_input(name, value=0.0, format="%.6f", key=f"feature_{name}")

    submitted = st.form_submit_button(
        "Run fraud-risk check",
        type="primary",
        use_container_width=True,
        disabled=not hybrid_ready,
    )

if not hybrid_ready:
    st.warning("Start the FastAPI server and confirm that /health reports hybrid_gnn as loaded. The check button will be enabled when ready.")

if submitted:
    if not card_id.strip():
        st.error("Please enter a card reference.")
    else:
        payload = {
            "card_id": card_id.strip(),
            "amount": float(amount),
            "features": features,
            "timestamp": time.time(),
        }
        try:
            with st.spinner("Checking transaction with the Autoencoder + GNN model..."):
                response = requests.post(EVALUATE_URL, json=payload, timeout=30)
            if response.ok:
                result = response.json()
                st.session_state.fg_last_result = result
                temporal = result.get("temporal_monitoring") or {}
                st.session_state.fg_log.append({
                    "Evaluation": len(st.session_state.fg_log) + 1,
                    "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Card reference": result.get("card_id", card_id.strip()),
                    "Amount (₹)": round(float(amount), 2),
                    "Decision": result.get("decision", "UNKNOWN"),
                    "Risk level": result.get("risk_level", "UNKNOWN"),
                    "Prototype score": float(result.get("risk_score", 0)),
                    "Transactions / min": int(temporal.get("velocity_1m", 0)),
                    "Transactions / hour": int(temporal.get("velocity_1h", 0)),
                    "Temporal alert": "; ".join(temporal.get("alerts") or []) or "None",
                })
                st.session_state.fg_log = st.session_state.fg_log[-100:]
                st.rerun()
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Request failed (HTTP {response.status_code}): {detail}")
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

result = st.session_state.fg_last_result
if result:
    st.divider()
    st.markdown("### 2. Your result")
    decision = str(result.get("decision", "UNKNOWN"))
    score = float(result.get("risk_score", 0.0))
    risk_level = str(result.get("risk_level", "UNKNOWN"))
    breakdown = result.get("breakdown") or {}
    temporal = result.get("temporal_monitoring") or {}
    alerts = temporal.get("alerts") or []

    result_col, detail_col = st.columns([1, 1.2], gap="large")
    with result_col:
        st.markdown('<div class="eyebrow">Model decision</div>', unsafe_allow_html=True)
        if decision == "APPROVE":
            st.success("APPROVE")
            explanation = "The prototype score is below its review threshold. This is a model suggestion, not a bank authorisation."
        elif decision == "OTP_VERIFY":
            st.warning("OTP VERIFY")
            explanation = "The prototype score reached its review threshold. A real system would request an additional identity check; this demo does not send an OTP."
        else:
            st.error(decision)
            explanation = "The API returned an unrecognised decision. Review the full response."
        st.markdown(f'<div class="result"><div class="eyebrow">Prototype fusion score · not a fraud probability</div><div style="font-size:2.5rem;font-weight:700;color:#f4f7ff">{score:.4f} / 1.0000</div></div>', unsafe_allow_html=True)
        st.progress(min(max(score, 0.0), 1.0))
        st.metric("Risk level returned by model", risk_level)
        st.write(explanation)
    with detail_col:
        st.markdown("**Why did the model return this?**")
        st.metric("GNN fraud score", f'{float(breakdown.get("gnn_fraud_score", 0)):.4f}')
        st.metric("Autoencoder reconstruction error", f'{float(breakdown.get("reconstruction_error", 0)):.4f}')
        st.caption(
            f"Decision threshold: {float(breakdown.get('fusion_threshold', 0.5)):.2f}. "
            "At or above this value the current prototype returns OTP_VERIFY; below it returns APPROVE."
        )
        conversion = result.get("amount_conversion") or {}
        st.markdown("**Amount used by the model**")
        st.write(f"Entered: ₹{float(conversion.get('amount_inr', amount)):,.2f}")
        st.caption(
            f"Converted with fixed project assumption ₹{float(conversion.get('inr_per_eur', 106.06)):.2f} per EUR. "
            "This is not a live FX rate and does not validate the model for Indian bank transactions."
        )

    st.markdown("### 3. Recent card activity")
    v1, v2, v3 = st.columns(3)
    v1.metric("Transactions in last minute", int(temporal.get("velocity_1m", 0)))
    v2.metric("Transactions in last hour", int(temporal.get("velocity_1h", 0)))
    v3.metric("History storage", str(temporal.get("history_backend", "unknown")).upper())
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No transaction-velocity alert was raised.")
    st.caption("Velocity alerts are separate from the model decision and do not currently change it.")

    with st.expander("Technical details (optional)"):
        st.write(f"**Graph method:** {breakdown.get('graph_method', 'Not reported')}")
        st.write(f"**Reference neighbours:** {breakdown.get('graph_neighbors', 'Not reported')}")
        st.write(f"**GNN validation threshold:** {float(breakdown.get('gnn_threshold', 0)):.4f}")
        st.json(result)

    st.warning(
        "Research prototype only. The fusion score is not a calibrated probability. "
        "The model uses anonymised demo features and has not been validated for real Indian banking transactions. "
        "Do not use this result to approve, decline, or authenticate a real payment."
    )

st.divider()
st.markdown("### Transaction history")
log = st.session_state.fg_log
if log:
    history = pd.DataFrame(log[::-1])
    chart = alt.Chart(history).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
        x=alt.X("Evaluation:O", title="Evaluation"),
        y=alt.Y("Prototype score:Q", title="Prototype score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Decision:N", scale=alt.Scale(domain=["APPROVE", "OTP_VERIFY"], range=["#36c99a", "#f0b84b"]), legend=alt.Legend(title="Decision")),
        tooltip=["Card reference:N", "Decision:N", "Risk level:N", "Prototype score:Q", "Amount (₹):Q"],
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button(
        "Download session history (CSV)",
        history.to_csv(index=False).encode("utf-8"),
        "fraudguard_session_history.csv",
        "text/csv",
    )
    st.caption("History is held in this Streamlit browser session. It is not a permanent transaction database.")
else:
    st.info("No checks yet. Complete the form above and select **Run fraud-risk check** to see your first result.")

st.caption("FraudGuard AI · Local research prototype · Autoencoder + feature-similarity GNN")
