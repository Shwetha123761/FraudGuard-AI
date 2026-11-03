import time
import requests
import pandas as pd
import streamlit as st
import altair as alt

API_URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{API_URL}/health"
EVALUATE_URL = f"{API_URL}/v1/evaluate-hybrid"

st.set_page_config(page_title="FraudGuard AI | Hybrid Risk Monitor", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {--bg:#080e1b;--panel:#101a2b;--line:#26354c;--text:#eaf1ff;--muted:#8fa1bb;--blue:#4ea1ff;--violet:#a879ff;--green:#35d6a1;--amber:#f5bd4f;--red:#ff6d87;}
.stApp {background:radial-gradient(ellipse at 15% 0%,rgba(45,104,205,.18),transparent 36%),radial-gradient(ellipse at 95% 10%,rgba(142,82,255,.12),transparent 30%),var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;}
[data-testid="stHeader"]{background:rgba(8,14,27,.85)} [data-testid="stSidebar"]{background:#0d1728;border-right:1px solid var(--line)} [data-testid="stSidebar"] *{color:var(--text)}
.block-container{max-width:1550px;padding-top:1.5rem;padding-bottom:2.5rem} h1,h2,h3,h4{font-family:'Space Grotesk',sans-serif!important;color:var(--text)!important}
p,label,.stMarkdown,[data-testid="stMetricLabel"]{color:var(--muted)} .brand{display:flex;align-items:center;gap:13px}
.brand-icon{width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#286de3,#8957ef);font-size:25px;box-shadow:0 0 26px rgba(78,161,255,.22)}
.eyebrow{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#73b7ff;font-weight:700} .title{font-family:'Space Grotesk',sans-serif;font-size:34px;line-height:1.1;font-weight:700;color:#f4f7ff;margin-top:3px}
.subtitle{font-size:13px;color:var(--muted);margin-top:6px} .section-label{font-size:10px;letter-spacing:1.8px;text-transform:uppercase;color:#7db9ff;font-weight:700;margin:7px 0 12px}
.card{background:linear-gradient(145deg,rgba(18,31,51,.98),rgba(13,23,39,.98));border:1px solid var(--line);border-radius:15px;padding:17px 18px;height:100%;box-shadow:0 10px 30px rgba(0,0,0,.10)}
.card-label{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);font-weight:700} .card-value{font-family:'Space Grotesk',sans-serif;font-size:26px;font-weight:700;color:#f4f7ff;margin-top:8px}
.card-foot{font-size:11px;color:var(--muted);margin-top:4px} .pill{display:inline-block;padding:5px 9px;border-radius:7px;font-size:10px;font-weight:700;letter-spacing:.4px}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:7px} .result{background:linear-gradient(135deg,rgba(42,102,220,.18),rgba(137,87,239,.14));border:1px solid rgba(91,151,255,.35);border-radius:16px;padding:22px}
.result-score{font-family:'Space Grotesk',sans-serif;font-size:38px;font-weight:700;color:#f4f7ff} div[data-testid="stButton"]>button{background:linear-gradient(100deg,#2874e8,#8058e9);color:white;border:1px solid #568df1;border-radius:9px;font-weight:700}
div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input{background:#0a1424;color:#eaf1ff;border-color:#2b3b54;border-radius:9px} [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:10px;overflow:hidden} hr{border-color:var(--line)}

/* Dashboard layout refresh */
.block-container{width:100%;max-width:1380px;padding-left:clamp(1rem,3vw,2.75rem);padding-right:clamp(1rem,3vw,2.75rem);padding-top:1.25rem}
.brand-icon{flex:0 0 48px}
.card{min-height:112px;transition:border-color .18s ease,transform .18s ease}
.card:hover{border-color:#45658f;transform:translateY(-1px)}
.section-label{margin-top:18px;margin-bottom:10px}
.result{min-height:218px;display:flex;flex-direction:column;justify-content:center;gap:5px}
.result-score{font-size:clamp(34px,3vw,44px);letter-spacing:-1.5px}
.score-track{height:9px;width:100%;background:#26354c;border-radius:99px;overflow:hidden;margin:10px 0 5px}
.score-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,#35d6a1,#f5bd4f,#ff6d87)}
[data-testid="stMetric"]{background:rgba(16,26,43,.72);border:1px solid #26354c;border-radius:12px;padding:12px 14px}
[data-testid="stMetricLabel"]{font-size:11px}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif}
div[data-testid="stDataFrame"]{width:100%}
@media(max-width:900px){
 .block-container{padding-left:1rem;padding-right:1rem}
 .title{font-size:28px}
 .card-value{font-size:23px}
}

</style>
""", unsafe_allow_html=True)

if "fg_log" not in st.session_state:
    st.session_state.fg_log = []
if "fg_last_result" not in st.session_state:
    st.session_state.fg_last_result = None

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-icon">🛡️</div><div><b style="font-size:19px">FraudGuard AI</b><div class="eyebrow">Hybrid risk monitor</div></div></div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Transaction simulator")
    card_id = st.text_input("Card ID", value="CARD_12345")
    # The API converts INR to the dataset's Amount scale before inference.
    # The training dataset's maximum original Amount was 25,691.16 (assumed EUR),
    # converted here using the API's fixed project rate of INR 106.06 per EUR.
    max_supported_inr = 25691.16 * 106.06
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        max_value=float(max_supported_inr),
        value=5000.0,
        step=100.0,
        format="%.2f",
        help=(
            "Enter the transaction amount in Indian rupees. The API converts it "
            "using a fixed project exchange-rate assumption before model inference. "
            "Amounts above the training dataset's original maximum are outside the supported range."
        ),
    )
    unusual_input = st.checkbox("Use unusual demo feature", value=False)
    st.caption("V1–V28 are generated demo inputs, not actual card history. Enter the transaction amount in INR (₹).")
    submit = st.button("⚡ Evaluate with hybrid model", use_container_width=True)
    st.divider()
    st.caption("Hybrid API endpoint")
    st.code("127.0.0.1:8000/v1/evaluate-hybrid", language=None)
    st.caption("Keep the FastAPI server running in its own terminal.")

health, health_error = None, None
try:
    hr = requests.get(HEALTH_URL, timeout=15)
    hr.raise_for_status()
    health = hr.json()
except requests.RequestException as exc:
    health_error = str(exc)

components = (health or {}).get("components", {})
ready = (health or {}).get("status") == "ready"
hybrid_ready = ready and components.get("hybrid_gnn") == "loaded"
api_color = "#35d6a1" if hybrid_ready else "#ff6d87"

st.markdown('<div class="brand"><div class="brand-icon">🛡️</div><div><div class="eyebrow">AI transaction intelligence</div><div class="title">FraudGuard <span style="color:#73b7ff">AI</span></div><div class="subtitle">Autoencoder anomaly detection · Feature-similarity GNN · Hybrid risk assessment</div></div></div>', unsafe_allow_html=True)
st.write("")
st.markdown(f'<span class="status-dot" style="background:{api_color};box-shadow:0 0 10px {api_color}"></span><span style="font-size:11px;letter-spacing:1.3px;font-weight:700;color:{api_color}">{"HYBRID MODEL READY" if hybrid_ready else "API OR HYBRID MODEL ISSUE"}</span><span style="color:#8fa1bb;font-size:11px"> · Local development environment</span>', unsafe_allow_html=True)
if not hybrid_ready:
    st.warning("The hybrid API is not ready. Check that Uvicorn is running and /health reports hybrid_gnn as loaded.")
    if health_error:
        st.caption(health_error)
    elif health:
        st.json(health)

log = st.session_state.fg_log
review_count = sum(1 for item in log if item["Decision"] != "APPROVE")
approval_rate = (100 * (len(log) - review_count) / len(log)) if log else None
st.markdown('<div class="section-label">System overview</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4, gap="medium")
cards = [
    ("SESSION TRANSACTIONS", str(len(log)), "Evaluations in this session", "#4ea1ff"),
    ("REVIEW FLAGS", str(review_count), "Non-approve decisions in session", "#a879ff"),
    ("APPROVAL RATE", f"{approval_rate:.1f}%" if approval_rate is not None else "—", "Based on this session" if log else "Awaiting first evaluation", "#35d6a1"),
    ("HYBRID PIPELINE", "READY" if hybrid_ready else "CHECK", "Autoencoder + feature-similarity GNN", "#f5bd4f"),
]
for col, (label, value, foot, color) in zip([m1, m2, m3, m4], cards):
    with col:
        st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value" style="color:{color}">{value}</div><div class="card-foot">{foot}</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.markdown('<div class="section-label">Transaction analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recent hybrid evaluations")
    if log:
        chart_df = pd.DataFrame(log[::-1])
        chart = alt.Chart(chart_df).mark_line(point=alt.OverlayMarkDef(size=65)).encode(
            x=alt.X("Evaluation:O", title="Recent evaluations", axis=alt.Axis(labelColor="#91a2bb", titleColor="#91a2bb", grid=False)),
            y=alt.Y("Fusion score:Q", title="Prototype fusion score", scale=alt.Scale(zero=True), axis=alt.Axis(labelColor="#91a2bb", titleColor="#91a2bb", gridColor="#26354c")),
            tooltip=["Card ID:N", "Decision:N", "Fusion score:Q", "Risk level:N"]
        ).properties(height=235).configure_view(strokeOpacity=0).configure(background="#101a2b").configure_axis(domainColor="#26354c")
        st.altair_chart(chart, use_container_width=True)

        st.markdown("**Transaction timeline · card velocity**")
        if "Transactions / 1 min" in chart_df.columns:
            velocity_df = chart_df[["Evaluation", "Time", "Card ID", "Transactions / 1 min", "Transactions / 1 hour"]].copy()
            velocity_long = velocity_df.melt(
                id_vars=["Evaluation", "Time", "Card ID"],
                value_vars=["Transactions / 1 min", "Transactions / 1 hour"],
                var_name="Window",
                value_name="Transaction count",
            )
            velocity_chart = alt.Chart(velocity_long).mark_line(point=True).encode(
                x=alt.X("Evaluation:O", title="Evaluation order", axis=alt.Axis(labelColor="#91a2bb", titleColor="#91a2bb", grid=False)),
                y=alt.Y("Transaction count:Q", title="Transactions counted", axis=alt.Axis(labelColor="#91a2bb", titleColor="#91a2bb", gridColor="#26354c")),
                color=alt.Color("Window:N", title="Time window"),
                tooltip=["Evaluation:O", "Time:N", "Card ID:N", "Window:N", "Transaction count:Q"],
            ).properties(height=200).configure_view(strokeOpacity=0).configure(background="#101a2b").configure_axis(domainColor="#26354c")
            st.altair_chart(velocity_chart, use_container_width=True)
    else:
        st.markdown('<div style="height:130px;display:flex;align-items:center;justify-content:center;color:#8fa1bb;text-align:center">No session data yet.<br>Evaluate a transaction to populate the chart.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">Detection components</div>', unsafe_allow_html=True)
    if health:
        rows = [
            ("Autoencoder", components.get("autoencoder", "unknown")),
            ("Supervised baseline (separate endpoint)", components.get("supervised_model", "unknown")),
            ("Fusion configuration (separate endpoint)", components.get("fusion_config", "unknown")),
            ("Hybrid GNN endpoint", components.get("hybrid_gnn", "unknown")),
        ]
        for name, status in rows:
            is_ready = status == "loaded"
            color = "#35d6a1" if is_ready else ("#f5bd4f" if status == "not_integrated" else "#ff6d87")
            st.markdown(f'<div class="card" style="padding:13px 15px;margin-bottom:9px;height:auto;display:flex;justify-content:space-between;align-items:center"><span style="color:#eaf1ff;font-size:13px;font-weight:600">{name}</span><span class="pill" style="background:{color}18;color:{color};border:1px solid {color}55">{status.replace("_"," ").upper()}</span></div>', unsafe_allow_html=True)
        st.caption("The dashboard calls the separate Autoencoder + feature-similarity GNN prototype endpoint.")
    else:
        st.warning("Model status unavailable.")

if submit:
    if not card_id.strip():
        st.error("Enter a Card ID.")
    elif not hybrid_ready:
        st.error("Cannot evaluate: the hybrid API is unavailable or not loaded.")
    else:
        features = {f"V{i}": 0.0 for i in range(1, 29)}
        if unusual_input:
            features["V1"], features["V2"] = 4.5, -3.0
        payload = {"card_id": card_id.strip(), "amount": float(amount), "features": features, "timestamp": time.time()}
        try:
            with st.spinner("Evaluating with the Autoencoder + GNN prototype..."):
                response = requests.post(EVALUATE_URL, json=payload, timeout=20)
            if response.ok:
                result = response.json()
                st.session_state.fg_last_result = result
                temporal = result.get("temporal_monitoring") or {}
                temporal_alerts = temporal.get("alerts") or []
                st.session_state.fg_log.append({
                    "Evaluation": len(st.session_state.fg_log) + 1,
                    "Time": time.strftime("%H:%M:%S"),
                    "Timestamp": float(result.get("timestamp", payload["timestamp"])),
                    "Card ID": result.get("card_id", card_id.strip()),
                    "Amount (INR)": round(float(amount), 2),
                    "Model-scaled Amount": round(float((result.get("amount_conversion") or {}).get("model_amount", 0.0)), 8),
                    "Decision": result.get("decision", "UNKNOWN"),
                    "Fusion score": float(result.get("risk_score", 0)),
                    "Risk level": result.get("risk_level", "UNKNOWN"),
                    "Transactions / 1 min": int(temporal.get("velocity_1m", 0)),
                    "Transactions / 1 hour": int(temporal.get("velocity_1h", 0)),
                    "Temporal alerts": "; ".join(temporal_alerts) if temporal_alerts else "None",
                })
                st.session_state.fg_log = st.session_state.fg_log[-100:]
                st.rerun()
            else:
                st.error(f"API returned HTTP {response.status_code}: {response.text}")
        except requests.RequestException as exc:
            st.error(f"Could not connect to FastAPI: {exc}")

result = st.session_state.fg_last_result
if result:
    st.write("")
    st.markdown('<div class="section-label">Latest hybrid transaction assessment</div>', unsafe_allow_html=True)
    decision = result.get("decision", "UNKNOWN")
    dcolor = {"APPROVE": "#35d6a1", "OTP_VERIFY": "#f5bd4f", "BLOCK": "#ff6d87"}.get(decision, "#91a1bb")
    breakdown = result.get("breakdown", {})
    temporal = result.get("temporal_monitoring") or {}
    st.markdown('<div class="section-label">Temporal monitoring · latest card activity</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    t1.metric("Transactions in last minute", int(temporal.get("velocity_1m", 0)))
    t2.metric("Transactions in last hour", int(temporal.get("velocity_1h", 0)))
    t3.metric("History backend", str(temporal.get("history_backend", "not reported")).upper())
    temporal_alerts = temporal.get("alerts") or []
    if temporal_alerts:
        for alert in temporal_alerts:
            st.warning(f"Temporal alert: {alert}")
    else:
        st.success("No temporal velocity alerts for this transaction.")
    st.caption(temporal.get(
        "note",
        "Temporal alerts are informational and do not currently change the hybrid model decision."
    ))
    a, b = st.columns([0.9, 1.1], gap="large")
    with a:
        fusion_score = float(result.get("risk_score", 0))
        score_width = min(max(fusion_score, 0.0), 1.0) * 100
        st.markdown(
            f'<div class="result">'
            f'<div class="card-label">HYBRID FUSION SCORE · PROTOTYPE</div>'
            f'<div class="result-score">{fusion_score:.6f}</div>'
            f'<div class="score-track"><div class="score-fill" style="width:{score_width:.2f}%"></div></div>'
            f'<div style="font-size:11px;color:#8fa1bb">Relative model score · not a calibrated fraud probability</div>'
            f'<div style="font-family:Space Grotesk,sans-serif;font-size:24px;font-weight:700;color:{dcolor};margin-top:12px">{decision.replace("_", " ")}</div>'
            f'<div style="color:#8fa1bb;font-size:12px;margin-top:3px">Risk level: {result.get("risk_level", "UNKNOWN")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with b:
        conversion = result.get("amount_conversion", {})
        if conversion:
            st.markdown("**Transaction amount conversion**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Amount entered", f"₹{float(conversion.get('amount_inr', 0)) :,.2f}")
            c2.metric("Assumed EUR equivalent", f"€{float(conversion.get('amount_eur_assumed', 0)) :,.2f}")
            c3.metric("Model Amount", f"{float(conversion.get('model_amount', 0)):.8f}")
            st.caption(
                f"Fixed project rate: ₹{float(conversion.get('inr_per_eur', 0)):.2f} per EUR. "
                "This is a project assumption, not a live exchange rate."
            )
        q1, q2, q3 = st.columns(3)
        q1.metric("GNN fraud score", f'{float(breakdown.get("gnn_fraud_score", 0)):.4f}')
        q2.metric("Reconstruction error", f'{float(breakdown.get("reconstruction_error", 0)):.4f}')
        q3.metric("Fusion threshold", f'{float(breakdown.get("fusion_threshold", 0)):.4f}')
        st.markdown("**Model details**")
        st.write(f"**Graph method:** {breakdown.get('graph_method', 'Not reported')}")
        st.write(f"**Reference neighbors:** {breakdown.get('graph_neighbors', 'Not reported')}")
        st.write(f"**GNN validation threshold:** {float(breakdown.get('gnn_threshold', 0)):.4f}")
        st.caption(result.get("note", ""))
        st.warning(
            "Research prototype: the hybrid model has been evaluated on the held-out test dataset "
            "(Precision: 62.50%, Recall: 73.33%, F1-score: 67.48%, ROC-AUC: 98.12%). "
            "The fusion score is not a calibrated fraud probability. These experimental results "
            "do not establish production readiness or real-world banking performance."
        )


st.write("")
st.markdown('<div class="section-label">Transaction history · current session</div>', unsafe_allow_html=True)
if st.session_state.fg_log:
    history = pd.DataFrame(st.session_state.fg_log[::-1])
    st.dataframe(history, use_container_width=True, hide_index=True)
    st.download_button("Export session history (CSV)", history.to_csv(index=False).encode("utf-8"), "fraudguard_hybrid_session_history.csv", "text/csv")
    st.caption("Timeline/history shows transactions evaluated in this Streamlit session. API velocity counts use the API process's in-memory history when Redis is unavailable.")
else:
    st.markdown('<div class="card"><div class="card-label">No transactions evaluated yet</div><div class="card-foot">Hybrid API results will appear here. No sample totals or fabricated transactions are displayed.</div></div>', unsafe_allow_html=True)

with st.expander("Inspect full API response"):
    if result:
        st.json(result)
    else:
        st.caption("Evaluate a transaction to view its full response.")

st.write("")
st.caption("FraudGuard AI · Local prototype · Autoencoder + feature-similarity GNN · Enter amounts in INR (₹) · Fusion score is not a calibrated probability")
