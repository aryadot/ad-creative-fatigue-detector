import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Ad Fatigue Detector",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
    }

    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }

    .metric-card h4 {
        color: #8b8fa8;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .metric-card h2 {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }

    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a2d3a;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        background-color: #1a1d27 !important;
        border: 1px solid #2a2d3a !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.5px;
        width: 100%;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.88;
    }

    label, .stSlider label {
        color: #9ca3af !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    .radar-container {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto 24px auto;
    }

    .radar-ring {
        position: absolute;
        border-radius: 50%;
        border: 1.5px solid #4f46e5;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        animation: radar-pulse 2.4s ease-out infinite;
        opacity: 0;
    }

    .radar-ring:nth-child(1) { animation-delay: 0s; }
    .radar-ring:nth-child(2) { animation-delay: 0.8s; }
    .radar-ring:nth-child(3) { animation-delay: 1.6s; }

    @keyframes radar-pulse {
        0%   { width: 12px; height: 12px; opacity: 0.9; }
        100% { width: 110px; height: 110px; opacity: 0; }
    }

    .radar-core {
        position: absolute;
        width: 12px; height: 12px;
        border-radius: 50%;
        background: #4f46e5;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0 12px #4f46e5;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model/xgb_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
explainer = load_explainer(model)

# ── Encoding maps ────────────────────────────────────────────
channel_map  = {"Facebook": 0, "Instagram": 1, "Pinterest": 2, "Twitter": 3}
audience_map = {"All Ages": 0, "Men 18-24": 1, "Men 25-34": 2, "Men 35-44": 3,
                "Men 45-60": 4, "Women 18-24": 5, "Women 25-34": 6,
                "Women 35-44": 7, "Women 45-60": 8}
goal_map     = {"Brand Awareness": 0, "Increase Sales": 1, "Market Expansion": 2, "Product Launch": 3}
segment_map  = {"Fashion": 0, "Food": 1, "Health": 2, "Home": 3, "Technology": 4}

# ── Header ───────────────────────────────────────────────────
st.markdown("<h1 style='color:#ffffff; font-size:32px; font-weight:700; margin-bottom:4px;'>Ad Creative Fatigue Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6b7280; font-size:15px; margin-bottom:32px;'>Powered by XGBoost + SHAP — input your campaign metrics to detect fatigue risk in real time.</p>", unsafe_allow_html=True)

# ── Layout ───────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='section-header'>Performance Metrics</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        clicks = st.number_input("Clicks", min_value=0, value=1000)
        engagement_score = st.number_input("Engagement Score", min_value=0, value=5)
        conversion_rate = st.number_input("Conversion Rate", min_value=0.0,
                                           max_value=1.0, value=0.05, step=0.01)
    with c2:
        impressions = st.number_input("Impressions", min_value=1, value=50000)
        acquisition_cost = st.number_input("Acquisition Cost ($)", min_value=0.0, value=25.0)
        duration_days = st.number_input("Duration (Days)", min_value=1, value=30)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Campaign Benchmarks</div>", unsafe_allow_html=True)
    ctr_vs_avg = st.slider("CTR vs Campaign Average", min_value=0.0, max_value=2.0,
                            value=1.0, step=0.01,
                            help="1.0 = at average. Below 0.85 signals decay.")
    engagement_vs_avg = st.slider("Engagement vs Campaign Average", min_value=0.0,
                                   max_value=2.0, value=1.0, step=0.01,
                                   help="1.0 = at average. Below 0.85 signals decay.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Campaign Details</div>", unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        channel = st.selectbox("Channel", list(channel_map.keys()))
        campaign_goal = st.selectbox("Campaign Goal", list(goal_map.keys()))
    with d2:
        target_audience = st.selectbox("Target Audience", list(audience_map.keys()))
        customer_segment = st.selectbox("Customer Segment", list(segment_map.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Analyze Campaign")

# ── Right panel ──────────────────────────────────────────────
with right:
    if not run:
        st.markdown("""
        <div style='height:420px; background:#1a1d27; border-radius:16px;
                    border: 1px solid #2a2d3a; padding-top:100px; text-align:center;'>
            <div class='radar-container'>
                <div class='radar-ring'></div>
                <div class='radar-ring'></div>
                <div class='radar-ring'></div>
                <div class='radar-core'></div>
            </div>
            <div style='color:#ffffff; font-size:16px; font-weight:600; margin-bottom:10px;'>
                No Analysis Yet
            </div>
            <div style='color:#6b7280; font-size:13px; max-width:220px;
                        margin:0 auto; line-height:1.7;'>
                Configure your campaign metrics on the left and click Analyze Campaign
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        ctr = clicks / impressions if impressions > 0 else 0

        input_data = pd.DataFrame([{
            'CTR': ctr,
            'CTR_vs_Campaign_Avg': ctr_vs_avg,
            'Engagement_Score': engagement_score,
            'Engagement_vs_Avg': engagement_vs_avg,
            'Conversion_Rate': conversion_rate,
            'Acquisition_Cost_Clean': acquisition_cost,
            'Duration_Days': duration_days,
            'Clicks': clicks,
            'Impressions': impressions,
            'Channel_Used': channel_map[channel],
            'Target_Audience': audience_map[target_audience],
            'Campaign_Goal': goal_map[campaign_goal],
            'Customer_Segment': segment_map[customer_segment]
        }])

        prob = model.predict_proba(input_data)[0][1]
        pred = 1 if prob >= 0.3 else 0

        # ── Result card ──
        st.markdown("<div class='section-header'>Detection Result</div>", unsafe_allow_html=True)

        if pred == 1:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#3d1515,#2a0f0f);
                        border:1px solid #c0392b; border-radius:16px;
                        padding:32px; text-align:center; margin-bottom:20px;'>
                <div style='font-size:13px; font-weight:600; letter-spacing:1px;
                            text-transform:uppercase; color:#e74c3c; margin-bottom:8px;'>
                    Fatigue Detected
                </div>
                <div style='font-size:64px; font-weight:800; color:#e74c3c; margin:12px 0;'>
                    {prob*100:.0f}%
                </div>
                <div style='font-size:14px; color:#a0a0a0; line-height:1.6;'>
                    This creative is showing fatigue signals. Consider refreshing your
                    creative, adjusting targeting, or pausing this campaign.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0f3d1e,#0a2a14);
                        border:1px solid #27ae60; border-radius:16px;
                        padding:32px; text-align:center; margin-bottom:20px;'>
                <div style='font-size:13px; font-weight:600; letter-spacing:1px;
                            text-transform:uppercase; color:#2ecc71; margin-bottom:8px;'>
                    Ad Is Healthy
                </div>
                <div style='font-size:64px; font-weight:800; color:#2ecc71; margin:12px 0;'>
                    {prob*100:.0f}%
                </div>
                <div style='font-size:14px; color:#a0a0a0; line-height:1.6;'>
                    Fatigue probability is within safe range.
                    Continue monitoring engagement and CTR trends closely.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Key metrics ──
        st.markdown("<div class='section-header'>Input Summary</div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class='metric-card'>
                <h4>CTR</h4><h2>{ctr*100:.2f}%</h2></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class='metric-card'>
                <h4>CTR vs Avg</h4><h2>{ctr_vs_avg:.2f}x</h2></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class='metric-card'>
                <h4>Eng vs Avg</h4><h2>{engagement_vs_avg:.2f}x</h2></div>""", unsafe_allow_html=True)

        # ── SHAP waterfall ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Feature Impact (SHAP)</div>", unsafe_allow_html=True)

        shap_vals = explainer.shap_values(input_data)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1d27')
        ax.set_facecolor('#1a1d27')

        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_data.iloc[0],
                feature_names=list(input_data.columns)
            ),
            show=False
        )

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('#9ca3af')
        ax.title.set_color('#ffffff')
        ax.xaxis.label.set_color('#9ca3af')

        plt.tight_layout()
        st.pyplot(fig)

# ── Footer ───────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<p style='color:#374151; font-size:12px; text-align:center;'>Ad Creative Fatigue Detector — XGBoost + SHAP — Marketing ML Portfolio</p>", unsafe_allow_html=True)