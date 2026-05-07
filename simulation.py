import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, time, os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Live Machine Simulation – ISE 298",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1B2A; color: #E8EEF7; }
    section[data-testid="stSidebar"] { background-color: #111E2E; }
    section[data-testid="stSidebar"] * { color: #E8EEF7 !important; }

    /* Metric cards */
    .card {
        background: #1A2B3C;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 5px solid #1565C0;
    }
    .card.green  { border-left-color: #2E7D32; }
    .card.amber  { border-left-color: #F57F17; }
    .card.red    { border-left-color: #C62828; }
    .card.purple { border-left-color: #6A1B9A; }
    .card.teal   { border-left-color: #00ACC1; }

    .card-title { font-size: 11px; color: #607D8B; font-weight:700;
                  text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
    .card-value { font-size: 30px; font-weight: 800; color: #FFFFFF; line-height:1.1; }
    .card-sub   { font-size: 12px; color: #546E7A; margin-top:4px; }

    /* Risk badge */
    .badge { display:inline-block; padding:5px 16px; border-radius:20px;
             font-weight:800; font-size:14px; letter-spacing:1px; }
    .badge-low      { background:#1B5E20; color:#A5D6A7; }
    .badge-medium   { background:#4E342E; color:#FFCC80; }
    .badge-high     { background:#4A0000; color:#EF9A9A; }
    .badge-critical { background:#880E4F; color:#FCE4EC; animation: blink 1s step-start infinite; }

    @keyframes blink { 50% { opacity: 0.4; } }

    /* Alert log */
    .alert-row {
        display:flex; justify-content:space-between; align-items:center;
        padding:8px 12px; border-radius:8px; margin-bottom:6px;
        font-size:12px; background:#1A2B3C;
    }
    .alert-actual   { color:#EF9A9A; font-weight:700; }
    .alert-predicted{ color:#80CBC4; font-weight:700; }
    .alert-correct  { color:#A5D6A7; }
    .alert-wrong    { color:#EF9A9A; }

    /* Status indicator */
    .status-dot {
        display:inline-block; width:10px; height:10px;
        border-radius:50%; margin-right:6px;
        animation: pulse 1.5s infinite;
    }
    .dot-green  { background:#4CAF50; }
    .dot-red    { background:#F44336; }
    .dot-grey   { background:#546E7A; animation:none; }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(76,175,80,0.6); }
        70%  { box-shadow: 0 0 0 8px rgba(76,175,80,0); }
        100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
    }

    /* Progress bar override */
    .stProgress > div > div { background-color: #1565C0; }

    /* Hide branding */
    #MainMenu { visibility:hidden; }
    footer     { visibility:hidden; }

    /* Section headers */
    .sec-hdr {
        font-size:14px; font-weight:700; color:#90A4AE;
        text-transform:uppercase; letter-spacing:1px;
        margin:16px 0 10px; border-bottom:1px solid #1A2B3C; padding-bottom:6px;
    }

    /* Sensor row */
    .sensor-row {
        display:flex; justify-content:space-between;
        padding:6px 10px; border-radius:6px;
        margin-bottom:4px; background:#1A2B3C; font-size:13px;
    }
    .sensor-name  { color:#90A4AE; }
    .sensor-value { color:#E8EEF7; font-weight:600; }
    .sensor-ok    { color:#4CAF50; }
    .sensor-warn  { color:#FF9800; }
    .sensor-crit  { color:#F44336; }
</style>
""", unsafe_allow_html=True)


# ── Load assets ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    base = os.path.dirname(os.path.abspath(__file__))
    sim  = pd.read_csv(os.path.join(base, "simulation_data.csv"))
    with open(os.path.join(base, "stats.json")) as f:
        stats = json.load(f)
    return sim, stats

sim_data, stats = load_assets()
TOTAL = len(sim_data)

FAILURE_MODES = {
    "TWF": ("Tool Wear Failure",        "#E53935"),
    "HDF": ("Heat Dissipation Failure", "#FB8C00"),
    "PWF": ("Power Failure",            "#8E24AA"),
    "OSF": ("Overstrain Failure",       "#1E88E5"),
    "RNF": ("Random Failure",           "#43A047"),
}

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running":      False,
        "current_idx":  0,
        "alert_log":    [],
        "prob_history": [],
        "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        "total_processed": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def reset_sim():
    st.session_state.running      = False
    st.session_state.current_idx  = 0
    st.session_state.alert_log    = []
    st.session_state.prob_history = []
    st.session_state.tp  = 0
    st.session_state.tn  = 0
    st.session_state.fp  = 0
    st.session_state.fn  = 0
    st.session_state.total_processed = 0


def risk_info(prob):
    if prob < 0.15: return "LOW",      "green",  "badge-low"
    if prob < 0.40: return "MEDIUM",   "amber",  "badge-medium"
    if prob < 0.70: return "HIGH",     "red",    "badge-high"
    return              "CRITICAL", "red",    "badge-critical"


def sensor_status(label, value):
    ranges = {
        "Air Temp":     (296, 304),
        "Process Temp": (306, 314),
        "RPM":          (1168, 2886),
        "Torque":       (3.8, 76.6),
        "Tool Wear":    (0, 200),
    }
    if label not in ranges:
        return "sensor-ok", "✅"
    lo, hi = ranges[label]
    if value < lo or value > hi:
        return "sensor-crit", "🔴"
    margin = (hi - lo) * 0.1
    if value > hi - margin or value < lo + margin:
        return "sensor-warn", "⚠️"
    return "sensor-ok", "✅"


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 Simulation Control Panel")
    st.markdown("Replaying **AI4I 2020** dataset as live machine data")
    st.markdown("---")

    # Speed
    speed = st.select_slider(
        "⚡ Simulation Speed",
        options=["0.25x", "0.5x", "1x", "2x", "5x", "10x", "Max"],
        value="1x",
        help="How fast to replay sensor data"
    )
    speed_map = {"0.25x": 4.0, "0.5x": 2.0, "1x": 1.0,
                 "2x": 0.5, "5x": 0.2, "10x": 0.1, "Max": 0.0}
    delay = speed_map[speed]

    # Threshold
    threshold = st.slider(
        "🎯 Alert Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Failure probability above this triggers an alert"
    )

    # Batch size
    batch = st.select_slider(
        "📦 Records per Step",
        options=[1, 5, 10, 25, 50],
        value=1,
        help="How many records to process per update"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Play" if not st.session_state.running else "⏸ Pause",
                     use_container_width=True, type="primary"):
            st.session_state.running = not st.session_state.running

    with col2:
        if st.button("⏹ Reset", use_container_width=True):
            reset_sim()
            st.rerun()

    # Jump to position
    st.markdown("---")
    st.markdown("#### 🎯 Jump to Record")
    jump_to = st.number_input("Record #", min_value=1,
                               max_value=TOTAL, value=1, step=1)
    if st.button("Go", use_container_width=True):
        reset_sim()
        st.session_state.current_idx = int(jump_to) - 1
        # Process up to jump point
        chunk = sim_data.iloc[:st.session_state.current_idx+1]
        st.session_state.prob_history = chunk["Failure Prob"].tolist()
        for _, r in chunk.iterrows():
            act = int(r["Actual Failure"])
            pred = 1 if r["Failure Prob"] >= threshold else 0
            if act==1 and pred==1: st.session_state.tp += 1
            elif act==0 and pred==0: st.session_state.tn += 1
            elif act==0 and pred==1: st.session_state.fp += 1
            else: st.session_state.fn += 1
        st.session_state.total_processed = len(chunk)
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📊 Dataset Info")
    st.markdown(f"""
- **Total records:** 10,000
- **Actual failures:** 339 (3.39%)
- **Features:** 5 sensors + type
- **Model:** Random Forest
- **Accuracy:** 98%
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:12px 0 4px;">
    <span style="font-size:22px; font-weight:800; color:#E8EEF7;">
        🏭 Live Machine Monitoring Simulation
    </span><br>
    <span style="font-size:13px; color:#546E7A;">
        ISE 298 · Replaying AI4I 2020 dataset in real-time · Random Forest Model
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Advance simulation one step ───────────────────────────────────────────────
if st.session_state.running and st.session_state.current_idx < TOTAL:
    end_idx = min(st.session_state.current_idx + batch, TOTAL)
    chunk   = sim_data.iloc[st.session_state.current_idx:end_idx]

    for _, row in chunk.iterrows():
        prob   = float(row["Failure Prob"])
        actual = int(row["Actual Failure"])
        pred   = 1 if prob >= threshold else 0

        st.session_state.prob_history.append(prob)

        # Confusion matrix counts
        if actual==1 and pred==1:   st.session_state.tp += 1
        elif actual==0 and pred==0: st.session_state.tn += 1
        elif actual==0 and pred==1: st.session_state.fp += 1
        else:                        st.session_state.fn += 1

        # Log alert if predicted failure OR actual failure
        if pred == 1 or actual == 1:
            fm_active = [fm for fm in ["TWF","HDF","PWF","OSF","RNF"] if row[fm]==1]
            fm_str    = ", ".join(fm_active) if fm_active else "Unknown"
            correct   = (pred == actual)
            st.session_state.alert_log.insert(0, {
                "Record":    int(row["UDI"]),
                "Prob":      f"{prob*100:.1f}%",
                "Predicted": "🔴 FAIL" if pred==1 else "🟢 OK",
                "Actual":    "🔴 FAIL" if actual==1 else "🟢 OK",
                "Match":     "✅" if correct else "❌",
                "Failure Mode": fm_str,
                "Torque":    f"{row['Torque']:.1f}",
                "RPM":       int(row["RPM"]),
                "Wear":      int(row["Tool Wear"]),
            })

    st.session_state.total_processed += len(chunk)
    st.session_state.current_idx = end_idx

    if st.session_state.current_idx >= TOTAL:
        st.session_state.running = False


# ── Get current row ────────────────────────────────────────────────────────────
idx = max(0, st.session_state.current_idx - 1)
row = sim_data.iloc[idx]
prob   = float(row["Failure Prob"])
actual = int(row["Actual Failure"])
level, card_color, badge_class = risk_info(prob)

# ── Status bar ─────────────────────────────────────────────────────────────────
processed = st.session_state.total_processed
progress  = processed / TOTAL
dot_class = "dot-green" if st.session_state.running else ("dot-red" if processed == TOTAL else "dot-grey")
status_txt = "RUNNING" if st.session_state.running else ("COMPLETED" if processed == TOTAL else "PAUSED")

st.markdown(f"""
<div style="display:flex; align-items:center; margin-bottom:8px;">
    <span class="status-dot {dot_class}"></span>
    <span style="font-size:13px; font-weight:700; color:#90A4AE; margin-right:16px;">
        {status_txt}
    </span>
    <span style="font-size:13px; color:#546E7A;">
        Record <b style="color:#E8EEF7;">{min(processed, TOTAL):,}</b> of
        <b style="color:#E8EEF7;">{TOTAL:,}</b>
        &nbsp;|&nbsp; Speed: <b style="color:#E8EEF7;">{speed}</b>
        &nbsp;|&nbsp; Alert threshold: <b style="color:#E8EEF7;">{threshold:.0%}</b>
    </span>
</div>
""", unsafe_allow_html=True)
st.progress(min(progress, 1.0))

st.markdown("<br>", unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="card {card_color}">
        <div class="card-title">Failure Probability</div>
        <div class="card-value">{prob*100:.1f}%</div>
        <div class="card-sub">Current reading</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card {card_color}">
        <div class="card-title">Risk Level</div>
        <div class="card-value" style="font-size:20px; padding-top:5px;">
            <span class="badge {badge_class}">{level}</span>
        </div>
        <div class="card-sub">Threshold: {threshold:.0%}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    act_str = "🔴 FAILURE" if actual==1 else "🟢 NORMAL"
    act_col = "#F44336" if actual==1 else "#4CAF50"
    st.markdown(f"""
    <div class="card {'red' if actual==1 else 'green'}">
        <div class="card-title">Actual Machine State</div>
        <div class="card-value" style="font-size:18px; padding-top:5px; color:{act_col};">
            {act_str}
        </div>
        <div class="card-sub">Ground truth label</div>
    </div>""", unsafe_allow_html=True)

with c4:
    tp = st.session_state.tp
    fp = st.session_state.fp
    fn = st.session_state.fn
    tn = st.session_state.tn
    total = tp+fp+fn+tn
    acc   = (tp+tn)/total*100 if total > 0 else 0
    st.markdown(f"""
    <div class="card teal">
        <div class="card-title">Live Accuracy</div>
        <div class="card-value">{acc:.1f}%</div>
        <div class="card-sub">{tp+tn:,} correct of {total:,}</div>
    </div>""", unsafe_allow_html=True)

with c5:
    alerts = st.session_state.tp + st.session_state.fp
    st.markdown(f"""
    <div class="card purple">
        <div class="card-title">Alerts Triggered</div>
        <div class="card-value">{alerts}</div>
        <div class="card-sub">{st.session_state.tp} true · {st.session_state.fp} false</div>
    </div>""", unsafe_allow_html=True)

# ── Main content row ──────────────────────────────────────────────────────────
left, mid, right = st.columns([1.1, 1.4, 1.1])

# ── LEFT — Current sensor readings ────────────────────────────────────────────
with left:
    st.markdown('<div class="sec-hdr">📡 Live Sensor Readings</div>', unsafe_allow_html=True)

    sensors = [
        ("Air Temp",     f"{row['Air Temp']:.1f} K",       row['Air Temp']),
        ("Process Temp", f"{row['Process Temp']:.1f} K",   row['Process Temp']),
        ("Temp Diff",    f"{row['Process Temp']-row['Air Temp']:.1f} K", row['Process Temp']-row['Air Temp']),
        ("RPM",          f"{int(row['RPM'])} rpm",          row['RPM']),
        ("Torque",       f"{row['Torque']:.1f} Nm",         row['Torque']),
        ("Tool Wear",    f"{int(row['Tool Wear'])} min",     row['Tool Wear']),
        ("Product Type", str(row['Type']),                   None),
    ]

    for name, val_str, val in sensors:
        css_class, icon = sensor_status(name, val) if val is not None else ("sensor-ok", "✅")
        st.markdown(f"""
        <div class="sensor-row">
            <span class="sensor-name">{icon} {name}</span>
            <span class="{css_class}">{val_str}</span>
        </div>""", unsafe_allow_html=True)

    # Failure mode indicators
    st.markdown('<div class="sec-hdr" style="margin-top:16px;">🔬 Active Failure Modes</div>',
                unsafe_allow_html=True)
    any_active = False
    for fm, (fm_name, color) in FAILURE_MODES.items():
        active = int(row[fm]) == 1
        if active:
            any_active = True
        dot = "🔴" if active else "⚪"
        weight = "800" if active else "400"
        opacity = "1" if active else "0.4"
        st.markdown(f"""
        <div class="sensor-row" style="opacity:{opacity};">
            <span class="sensor-name" style="font-weight:{weight};">{dot} {fm}</span>
            <span style="color:{color}; font-size:11px;">{fm_name}</span>
        </div>""", unsafe_allow_html=True)

    if not any_active:
        st.markdown('<div style="font-size:12px; color:#546E7A; padding:4px 10px;">No active failure modes</div>',
                    unsafe_allow_html=True)

# ── MID — Live probability chart ──────────────────────────────────────────────
with mid:
    st.markdown('<div class="sec-hdr">📈 Failure Probability — Live Chart</div>',
                unsafe_allow_html=True)

    history = st.session_state.prob_history
    if len(history) > 0:
        # Show last 200 points
        window   = history[-200:]
        chart_df = pd.DataFrame({
            "Record": range(max(0, len(history)-200), len(history)),
            "Failure Probability": window,
            "Alert Threshold":     [threshold] * len(window),
        }).set_index("Record")
        st.line_chart(chart_df, color=["#1E88E5", "#F44336"], height=220)

        # Distribution of probabilities so far
        st.markdown('<div class="sec-hdr">📊 Probability Distribution</div>',
                    unsafe_allow_html=True)
        bins = [0, 0.15, 0.40, 0.70, 1.01]
        labels = ["LOW\n(<15%)", "MEDIUM\n(15-40%)", "HIGH\n(40-70%)", "CRITICAL\n(>70%)"]
        colors_list = ["#2E7D32", "#F57F17", "#C62828", "#880E4F"]
        counts = pd.cut(history, bins=bins, labels=labels).value_counts()

        dist_df = pd.DataFrame({
            "Risk Level": labels,
            "Count": [counts.get(l, 0) for l in labels]
        }).set_index("Risk Level")
        st.bar_chart(dist_df, color="#1E88E5", height=160)
    else:
        st.markdown("""
        <div style="height:200px; display:flex; align-items:center; justify-content:center;
                    color:#546E7A; font-size:14px; background:#1A2B3C; border-radius:8px;">
            ▶️ Press Play to start the simulation
        </div>""", unsafe_allow_html=True)

# ── RIGHT — Accuracy metrics & alert log ──────────────────────────────────────
with right:
    st.markdown('<div class="sec-hdr">🎯 Model Performance (Live)</div>',
                unsafe_allow_html=True)

    tp = st.session_state.tp
    fp = st.session_state.fp
    fn = st.session_state.fn
    tn = st.session_state.tn
    total = tp + fp + fn + tn

    precision = tp/(tp+fp)*100 if (tp+fp)>0 else 0
    recall    = tp/(tp+fn)*100 if (tp+fn)>0 else 0
    f1        = 2*(precision*recall)/(precision+recall) if (precision+recall)>0 else 0
    acc       = (tp+tn)/total*100 if total>0 else 0

    metrics = [
        ("Accuracy",  f"{acc:.1f}%",       "teal"),
        ("Precision", f"{precision:.1f}%",  "green"),
        ("Recall",    f"{recall:.1f}%",     "purple"),
        ("F1 Score",  f"{f1:.1f}%",         "amber"),
    ]
    for title, val, col in metrics:
        st.markdown(f"""
        <div class="card {col}" style="padding:10px 16px; margin-bottom:8px;">
            <div class="card-title">{title}</div>
            <div class="card-value" style="font-size:22px;">{val}</div>
        </div>""", unsafe_allow_html=True)

    # Confusion matrix
    st.markdown('<div class="sec-hdr">🧮 Confusion Matrix</div>', unsafe_allow_html=True)
    cm_df = pd.DataFrame({
        "Predicted OK":   [f"TN: {tn:,}", f"FN: {fn:,}"],
        "Predicted FAIL": [f"FP: {fp:,}", f"TP: {tp:,}"],
    }, index=["Actual OK", "Actual FAIL"])
    st.dataframe(cm_df, use_container_width=True)


# ── Alert Log ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec-hdr">🚨 Alert Log — Failure Events Detected</div>',
            unsafe_allow_html=True)

alerts = st.session_state.alert_log
if alerts:
    log_df = pd.DataFrame(alerts[:50])  # show last 50
    st.dataframe(
        log_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Record":       st.column_config.NumberColumn("Record #", width="small"),
            "Prob":         st.column_config.TextColumn("Fail Prob", width="small"),
            "Predicted":    st.column_config.TextColumn("Predicted", width="small"),
            "Actual":       st.column_config.TextColumn("Actual", width="small"),
            "Match":        st.column_config.TextColumn("✓", width="small"),
            "Failure Mode": st.column_config.TextColumn("Failure Mode"),
            "Torque":       st.column_config.TextColumn("Torque", width="small"),
            "RPM":          st.column_config.NumberColumn("RPM", width="small"),
            "Wear":         st.column_config.NumberColumn("Wear(min)", width="small"),
        }
    )
    tp_count = sum(1 for a in alerts if a["Match"]=="✅" and a["Actual"]=="🔴 FAIL")
    fp_count = sum(1 for a in alerts if a["Match"]=="❌")
    fn_count = st.session_state.fn
    st.caption(f"Showing last {min(50, len(alerts))} events | "
               f"✅ Correct: {tp_count} | ❌ Wrong: {fp_count} | "
               f"Missed failures: {fn_count}")
else:
    st.markdown("""
    <div style="padding:20px; text-align:center; color:#546E7A; font-size:14px;
                background:#1A2B3C; border-radius:8px;">
        No alerts yet — press ▶️ Play to start simulation
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#37474F; font-size:11px;">
    ISE 298 Graduate Project · SJSU · Spring 2025 &nbsp;|&nbsp;
    AI4I 2020 Dataset (UCI) — 10,000 records &nbsp;|&nbsp;
    Random Forest · VIKOR Q=0.012 · 98% Accuracy
</div>
""", unsafe_allow_html=True)

# ── Auto-refresh while running ────────────────────────────────────────────────
if st.session_state.running and st.session_state.current_idx < TOTAL:
    time.sleep(delay)
    st.rerun()
