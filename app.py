# ============================================================
# TELECOM CUSTOMER CHURN PREDICTION SYSTEM
# Streamlit Web Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title = "Telecom Churn Prediction",
    page_icon  = "📱",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .red-card {
        background: linear-gradient(135deg, #FF4444, #CC0000);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255,68,68,0.4);
    }
    .yellow-card {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255,215,0,0.4);
    }
    .green-card {
        background: linear-gradient(135deg, #00C851, #007E33);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,200,81,0.4);
    }
    .metric-card {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 0.5rem 0;
    }
    .strategy-box {
        background: #E8F4FD;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2E86AB;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL AND FILES
# ============================================================
@st.cache_resource
def load_model_files():
    """Load all saved model files"""
    try:
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))

        model     = joblib.load(
            os.path.join(BASE_PATH, 'models',
                         'final_xgboost_model.pkl')
        )
        scaler    = joblib.load(
            os.path.join(BASE_PATH, 'models',
                         'scaler.pkl')
        )
        explainer = joblib.load(
            os.path.join(BASE_PATH, 'models',
                         'shap_explainer.pkl')
        )

        with open(os.path.join(
            BASE_PATH, 'models', 'feature_names.json'
        ), 'r') as f:
            feature_names = json.load(f)

        with open(os.path.join(
            BASE_PATH, 'results', 'project_summary.json'
        ), 'r') as f:
            project_summary = json.load(f)

        shap_imp = pd.read_csv(
            os.path.join(BASE_PATH, 'results',
                         'shap_importance.csv')
        )
        model_results = pd.read_csv(
            os.path.join(BASE_PATH, 'results',
                         'model_results.csv')
        )

        return (model, scaler, explainer,
                feature_names, project_summary,
                shap_imp, model_results, True)

    except Exception as e:
        return (None, None, None, None,
                None, None, None, False)

# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================
def engineer_features(df):
    """Apply same feature engineering as training"""

    df = df.copy()

    # ARPU features
    if all(c in df.columns
           for c in ['arpu_6','arpu_7','arpu_8']):
        df['arpu_trend']     = df['arpu_8'] - df['arpu_6']
        df['arpu_trend_7_8'] = df['arpu_8'] - df['arpu_7']
        df['arpu_decline_rate'] = (
            (df['arpu_6'] - df['arpu_8']) /
            (df['arpu_6'] + 1)
        )
        df['avg_arpu'] = df[
            ['arpu_6','arpu_7','arpu_8']
        ].mean(axis=1)
        df['arpu_std'] = df[
            ['arpu_6','arpu_7','arpu_8']
        ].std(axis=1)

    # Call usage features
    if all(c in df.columns
           for c in ['total_og_mou_6',
                     'total_og_mou_8']):
        df['og_mou_trend'] = (
            df['total_og_mou_8'] -
            df['total_og_mou_6']
        )
        df['aug_jun_og_ratio'] = (
            (df['total_og_mou_8'] + 1) /
            (df['total_og_mou_6'] + 1)
        )

    if all(c in df.columns
           for c in ['total_ic_mou_6',
                     'total_ic_mou_8']):
        df['ic_mou_trend'] = (
            df['total_ic_mou_8'] -
            df['total_ic_mou_6']
        )

    # Average calls
    og_cols = [c for c in df.columns
               if 'total_og_mou' in c]
    if og_cols:
        df['avg_og_mou'] = df[og_cols].mean(axis=1)

    ic_cols = [c for c in df.columns
               if 'total_ic_mou' in c]
    if ic_cols:
        df['avg_ic_mou'] = df[ic_cols].mean(axis=1)

    # Total mou per month
    for month in ['6', '7', '8']:
        og = f'total_og_mou_{month}'
        ic = f'total_ic_mou_{month}'
        if og in df.columns and ic in df.columns:
            df[f'total_mou_{month}'] = (
                df[og] + df[ic]
            )

    # OG std
    og_std_cols = [
        'total_og_mou_6',
        'total_og_mou_7',
        'total_og_mou_8'
    ]
    if all(c in df.columns for c in og_std_cols):
        df['og_mou_std'] = df[og_std_cols].std(axis=1)

    # Recharge features
    if all(c in df.columns
           for c in ['total_rech_amt_6',
                     'total_rech_amt_8']):
        df['rech_amt_trend'] = (
            df['total_rech_amt_8'] -
            df['total_rech_amt_6']
        )

    rech_cols = [c for c in df.columns
                 if 'total_rech_amt' in c]
    if rech_cols:
        df['avg_rech_amt'] = df[rech_cols].mean(axis=1)

    if all(c in df.columns
           for c in ['total_rech_num_6',
                     'total_rech_num_8']):
        df['rech_num_trend'] = (
            df['total_rech_num_8'] -
            df['total_rech_num_6']
        )

    # Customer profile
    if 'aon' in df.columns:
        df['tenure_months'] = (
            df['aon'] / 30
        ).round(0)
        df['tenure_segment'] = pd.cut(
            df['tenure_months'],
            bins   = [0, 6, 12, 24, 36, 10000],
            labels = [5, 4, 3, 2, 1]
        ).astype(float).fillna(3)

    if 'avg_arpu' in df.columns:
        q75 = 356.23
        df['is_high_value'] = (
            df['avg_arpu'] >= q75
        ).astype(int)

    roam_cols = [c for c in df.columns
                 if 'roam' in c.lower()]
    if roam_cols:
        df['is_roaming_user'] = (
            df[roam_cols].sum(axis=1) > 0
        ).astype(int)

    data_rech_cols = [
        c for c in df.columns
        if 'av_rech_amt_data' in c
    ]
    if data_rech_cols:
        df['avg_data_rech'] = (
            df[data_rech_cols].mean(axis=1)
        )

    return df

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_churn(df, model, scaler, feature_names):
    """Predict churn for given dataframe"""

    # Feature engineering
    df_eng = engineer_features(df)

    # Fill missing
    df_eng = df_eng.fillna(0)
    df_eng = df_eng.replace(
        [np.inf, -np.inf], 0
    )

    # Add missing columns with 0
    for col in feature_names:
        if col not in df_eng.columns:
            df_eng[col] = 0

    # Reorder features
    X = df_eng[feature_names]
    X = X.astype(np.float64)

    # Scale
    X_scaled = scaler.transform(X.values)

    # Predict
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = model.predict(X_scaled)

    return probs, preds, X_scaled

# ============================================================
# RISK CATEGORY FUNCTION
# ============================================================
def get_risk_category(prob):
    if prob >= 0.70:
        return 'RED', '🔴', 'HIGH RISK'
    elif prob >= 0.40:
        return 'YELLOW', '🟡', 'MEDIUM RISK'
    else:
        return 'GREEN', '🟢', 'LOW RISK'

# ============================================================
# RETENTION STRATEGY FUNCTION
# ============================================================
def get_strategy(row_dict, prob, risk):
    reasons  = []
    strategy = []

    arpu_trend     = row_dict.get('arpu_trend', 0)
    og_mou_trend   = row_dict.get('og_mou_trend', 0)
    rech_amt_trend = row_dict.get('rech_amt_trend', 0)
    rech_num_trend = row_dict.get('rech_num_trend', 0)
    tenure_months  = row_dict.get('tenure_months', 12)
    is_high_value  = row_dict.get('is_high_value', 0)
    is_roaming     = row_dict.get('is_roaming_user', 0)

    if arpu_trend < -100:
        reasons.append("Revenue dropped significantly")
        strategy.append("💰 Offer budget-friendly plan")
    elif arpu_trend < -50:
        reasons.append("Revenue slightly declining")
        strategy.append("💰 Small discount on current plan")

    if og_mou_trend < -100:
        reasons.append("Call usage dropped heavily")
        strategy.append("📱 Offer unlimited calling bundle")
    elif og_mou_trend < -50:
        reasons.append("Call usage declining")
        strategy.append("📱 Discounted calling pack")

    if rech_amt_trend < -100:
        reasons.append("Recharge amount dropped")
        strategy.append("🎁 Auto-recharge cashback offer")
    elif rech_amt_trend < -50:
        reasons.append("Recharge declining")
        strategy.append("🎁 Bonus data on next recharge")

    if rech_num_trend < -3:
        reasons.append("Recharging less frequently")
        strategy.append("🔔 Recharge reminder + cashback")

    if tenure_months < 6:
        reasons.append("New customer (< 6 months)")
        strategy.append("👋 Welcome bonus: extra data")
    elif tenure_months > 36:
        reasons.append("Long-term loyal customer at risk")
        strategy.append("👑 VIP loyalty reward")
    elif tenure_months > 24:
        reasons.append("2+ year customer at risk")
        strategy.append("⭐ Loyalty discount: 20% off")

    if is_high_value == 1:
        reasons.append("High value customer")
        strategy.append("💎 Dedicated relationship manager")

    if is_roaming == 1:
        reasons.append("Active roaming user")
        strategy.append("✈️ Special roaming pack")

    if not reasons:
        if risk == 'RED':
            reasons  = ["High churn probability"]
            strategy = [
                "🔴 Emergency retention call",
                "💎 Best available plan",
                "🎁 3 months at 50% price"
            ]
        elif risk == 'YELLOW':
            reasons  = ["Moderate churn risk"]
            strategy = [
                "📱 Plan upgrade suggestion",
                "💰 Loyalty discount"
            ]
        else:
            reasons  = ["Customer is stable"]
            strategy = [
                "✅ Regular service",
                "🎁 Loyalty points"
            ]

    if risk == 'RED':
        action   = "🚨 IMMEDIATE call from retention team"
        timeline = "Within 24 hours"
        offer    = "Up to 30% discount OR 2 months free"
    elif risk == 'YELLOW':
        action   = "📧 Personalized SMS + App notification"
        timeline = "Within 1 week"
        offer    = "10-15% discount OR bonus data"
    else:
        action   = "📊 Regular engagement program"
        timeline = "Monthly newsletter"
        offer    = "Loyalty reward points"

    return reasons, strategy, action, timeline, offer

# ============================================================
# SHAP EXPLANATION FUNCTION
# ============================================================
def get_shap_explanation(
    explainer, X_scaled, feature_names, top_n=10
):
    try:
        shap_vals = explainer.shap_values(X_scaled)
        if len(shap_vals.shape) == 1:
            sv = shap_vals
        else:
            sv = shap_vals[0]

        shap_df = pd.DataFrame({
            'Feature'   : feature_names,
            'SHAP_Value': sv,
            'Abs_SHAP'  : np.abs(sv)
        }).sort_values(
            'Abs_SHAP', ascending=False
        ).head(top_n)

        return shap_df
    except:
        return None

# ============================================================
# MAIN APP
# ============================================================
def main():

    # Load files
    (model, scaler, explainer,
     feature_names, project_summary,
     shap_imp, model_results,
     loaded) = load_model_files()

    # ── Header ────────────────────────────────────────────────
    st.markdown(
        '<div class="main-header">'
        '📱 Telecom Customer Churn Prediction System'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">'
        'Powered by XGBoost + SHAP Explainability | '
        'Indian Telecom Dataset'
        '</div>',
        unsafe_allow_html=True
    )

    if not loaded:
        st.error(
            "❌ Model files not found! "
            "Please check models/ folder."
        )
        return

    st.success("✅ Model loaded successfully!")

    # ── Sidebar ───────────────────────────────────────────────
    st.sidebar.image(
        "https://img.icons8.com/color/96/phone.png",
        width=80
    )
    st.sidebar.title("🎯 Navigation")

    page = st.sidebar.radio(
        "Choose Mode:",
        [
            "🏠 Home / Dashboard",
            "👤 Single Customer Prediction",
            "📁 Bulk CSV Prediction",
            "📊 Model Performance"
        ]
    )

    # ── Sidebar Stats ─────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Stats")
    if project_summary:
        st.sidebar.metric(
            "Accuracy",
            f"{project_summary['Best_Accuracy']*100:.1f}%"
        )
        st.sidebar.metric(
            "AUC-ROC",
            f"{project_summary['Best_AUC']*100:.1f}%"
        )
        st.sidebar.metric(
            "Total Customers",
            f"{project_summary['Total_Customers']:,}"
        )

    # ══════════════════════════════════════════════════════════
    # PAGE 1: HOME DASHBOARD
    # ══════════════════════════════════════════════════════════
    if page == "🏠 Home / Dashboard":

        st.markdown("## 🏠 Project Dashboard")

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "🎯 Accuracy",
                f"{project_summary['Best_Accuracy']*100:.1f}%"
            )
        with col2:
            st.metric(
                "📈 AUC-ROC",
                f"{project_summary['Best_AUC']*100:.1f}%"
            )
        with col3:
            st.metric(
                "🎯 Precision",
                f"{project_summary['Best_Precision']*100:.1f}%"
            )
        with col4:
            st.metric(
                "🔍 Recall",
                f"{project_summary['Best_Recall']*100:.1f}%"
            )
        with col5:
            st.metric(
                "⚖️ F1-Score",
                f"{project_summary['Best_F1']*100:.1f}%"
            )

        st.markdown("---")

        # Risk distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Risk Segmentation")

            red_c    = project_summary['RED_Customers']
            yellow_c = project_summary['YELLOW_Customers']
            green_c  = project_summary['GREEN_Customers']

            fig = go.Figure(data=[go.Pie(
                labels = ['🔴 RED', '🟡 YELLOW', '🟢 GREEN'],
                values = [red_c, yellow_c, green_c],
                hole   = 0.4,
                marker = dict(
                    colors = ['#FF4444', '#FFD700', '#00C851']
                )
            )])
            fig.update_layout(
                height = 350,
                showlegend = True,
                title  = "Customer Risk Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📊 Risk Validation")

            categories   = ['🔴 RED', '🟡 YELLOW', '🟢 GREEN']
            churn_rates  = [93.9, 61.8, 1.4]
            colors_chart = ['#FF4444', '#FFD700', '#00C851']

            fig2 = go.Figure(data=[
                go.Bar(
                    x      = categories,
                    y      = churn_rates,
                    marker_color = colors_chart,
                    text   = [f'{v}%' for v in churn_rates],
                    textposition = 'outside'
                )
            ])
            fig2.update_layout(
                height     = 350,
                title      = "Actual Churn Rate per Category",
                yaxis_title= "Actual Churn Rate (%)",
                yaxis      = dict(range=[0, 115])
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # SHAP importance
        st.markdown("### 🔍 Top 15 Important Features (SHAP)")

        top15 = shap_imp.head(15).sort_values(
            'SHAP_Value', ascending=True
        )
        fig3 = go.Figure(go.Bar(
            x           = top15['SHAP_Value'],
            y           = top15['Feature'],
            orientation = 'h',
            marker_color= '#2E86AB'
        ))
        fig3.update_layout(
            height      = 500,
            title       = "Feature Importance by SHAP",
            xaxis_title = "Mean |SHAP Value|"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Model comparison
        st.markdown("### 📊 Model Comparison")
        st.dataframe(
            model_results[[
                'Model', 'Accuracy', 'Precision',
                'Recall', 'F1-Score', 'AUC-ROC'
            ]].style.highlight_max(
                subset=['Accuracy', 'Precision',
                        'F1-Score', 'AUC-ROC'],
                color='lightgreen'
            ),
            use_container_width=True
        )

    # ══════════════════════════════════════════════════════════
    # PAGE 2: SINGLE CUSTOMER PREDICTION
    # ══════════════════════════════════════════════════════════
    elif page == "👤 Single Customer Prediction":

        st.markdown("## 👤 Single Customer Prediction")
        st.info(
            "Enter customer details below. "
            "All other features will use default values."
        )

        # Input form
        with st.form("customer_form"):
            st.markdown("### 📊 Revenue (ARPU)")
            col1, col2, col3 = st.columns(3)
            with col1:
                arpu_6 = st.number_input(
                    "ARPU June (₹)",
                    min_value=0.0,
                    value=280.0,
                    step=10.0
                )
            with col2:
                arpu_7 = st.number_input(
                    "ARPU July (₹)",
                    min_value=0.0,
                    value=250.0,
                    step=10.0
                )
            with col3:
                arpu_8 = st.number_input(
                    "ARPU August (₹)",
                    min_value=0.0,
                    value=100.0,
                    step=10.0
                )

            st.markdown("### 📞 Outgoing Calls (minutes)")
            col1, col2, col3 = st.columns(3)
            with col1:
                og_6 = st.number_input(
                    "Outgoing June (min)",
                    min_value=0.0,
                    value=300.0,
                    step=10.0
                )
            with col2:
                og_7 = st.number_input(
                    "Outgoing July (min)",
                    min_value=0.0,
                    value=200.0,
                    step=10.0
                )
            with col3:
                og_8 = st.number_input(
                    "Outgoing August (min)",
                    min_value=0.0,
                    value=50.0,
                    step=10.0
                )

            st.markdown("### 📲 Incoming Calls (minutes)")
            col1, col2, col3 = st.columns(3)
            with col1:
                ic_6 = st.number_input(
                    "Incoming June (min)",
                    min_value=0.0,
                    value=250.0,
                    step=10.0
                )
            with col2:
                ic_7 = st.number_input(
                    "Incoming July (min)",
                    min_value=0.0,
                    value=180.0,
                    step=10.0
                )
            with col3:
                ic_8 = st.number_input(
                    "Incoming August (min)",
                    min_value=0.0,
                    value=30.0,
                    step=10.0
                )

            st.markdown("### 💳 Recharge Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                rech_amt_6 = st.number_input(
                    "Recharge Amount June (₹)",
                    min_value=0.0,
                    value=300.0,
                    step=10.0
                )
            with col2:
                rech_amt_7 = st.number_input(
                    "Recharge Amount July (₹)",
                    min_value=0.0,
                    value=200.0,
                    step=10.0
                )
            with col3:
                rech_amt_8 = st.number_input(
                    "Recharge Amount August (₹)",
                    min_value=0.0,
                    value=50.0,
                    step=10.0
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                rech_num_6 = st.number_input(
                    "Recharge Count June",
                    min_value=0,
                    value=4,
                    step=1
                )
            with col2:
                rech_num_7 = st.number_input(
                    "Recharge Count July",
                    min_value=0,
                    value=3,
                    step=1
                )
            with col3:
                rech_num_8 = st.number_input(
                    "Recharge Count August",
                    min_value=0,
                    value=1,
                    step=1
                )

            st.markdown("### 👤 Customer Profile")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                aon = st.number_input(
                    "Age on Network (days)",
                    min_value=0,
                    value=365,
                    step=30
                )
            with col2:
                last_rch = st.number_input(
                    "Last Recharge Amount (₹)",
                    min_value=0.0,
                    value=50.0,
                    step=10.0
                )
            with col3:
                roam_og = st.number_input(
                    "Roaming Outgoing (min)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0
                )
            with col4:
                roam_ic = st.number_input(
                    "Roaming Incoming (min)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0
                )

            submitted = st.form_submit_button(
                "🔮 PREDICT CHURN",
                use_container_width=True
            )

        # ── Prediction ────────────────────────────────────────
        if submitted:
            # Build customer dataframe
            customer_data = {
                'arpu_6'           : arpu_6,
                'arpu_7'           : arpu_7,
                'arpu_8'           : arpu_8,
                'total_og_mou_6'   : og_6,
                'total_og_mou_7'   : og_7,
                'total_og_mou_8'   : og_8,
                'total_ic_mou_6'   : ic_6,
                'total_ic_mou_7'   : ic_7,
                'total_ic_mou_8'   : ic_8,
                'total_rech_amt_6' : rech_amt_6,
                'total_rech_amt_7' : rech_amt_7,
                'total_rech_amt_8' : rech_amt_8,
                'total_rech_num_6' : rech_num_6,
                'total_rech_num_7' : rech_num_7,
                'total_rech_num_8' : rech_num_8,
                'aon'              : aon,
                'last_day_rch_amt_8': last_rch,
                'roam_og_mou_8'    : roam_og,
                'roam_ic_mou_8'    : roam_ic,
            }

            # Add all other features as 0
            all_base_cols = feature_names.copy()
            for col in all_base_cols:
                if col not in customer_data:
                    customer_data[col] = 0

            df_customer = pd.DataFrame([customer_data])

            # Predict
            with st.spinner("🔮 Predicting..."):
                probs, preds, X_scaled = predict_churn(
                    df_customer, model,
                    scaler, feature_names
                )

            prob = probs[0]
            risk, emoji, risk_msg = get_risk_category(prob)

            # ── Risk Card ─────────────────────────────────────
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if risk == 'RED':
                    st.markdown(
                        f'<div class="red-card">'
                        f'{emoji} {risk_msg}<br>'
                        f'Churn Probability: {prob*100:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                elif risk == 'YELLOW':
                    st.markdown(
                        f'<div class="yellow-card">'
                        f'{emoji} {risk_msg}<br>'
                        f'Churn Probability: {prob*100:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="green-card">'
                        f'{emoji} {risk_msg}<br>'
                        f'Churn Probability: {prob*100:.1f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # ── Gauge Chart ───────────────────────────────────
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prob * 100,
                title = {'text': "Churn Probability (%)"},
                gauge = {
                    'axis' : {'range': [0, 100]},
                    'bar'  : {'color': (
                        '#FF4444' if risk == 'RED'
                        else '#FFD700' if risk == 'YELLOW'
                        else '#00C851'
                    )},
                    'steps': [
                        {'range': [0, 40],
                         'color': '#E8F5E9'},
                        {'range': [40, 70],
                         'color': '#FFF9C4'},
                        {'range': [70, 100],
                         'color': '#FFEBEE'}
                    ],
                    'threshold': {
                        'line' : {'color': 'red',
                                  'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(
                fig_gauge, use_container_width=True
            )

            # ── Get engineered features ───────────────────────
            df_eng = engineer_features(df_customer)
            df_eng = df_eng.fillna(0)
            row_dict = df_eng.iloc[0].to_dict()

            # ── Retention Strategy ────────────────────────────
            (reasons, strategy,
             action, timeline, offer) = get_strategy(
                row_dict, prob, risk
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ⚠️ Why Customer May Churn")
                for r in reasons:
                    st.markdown(f"- {r}")

                st.markdown("### 💡 Retention Strategies")
                for s in strategy:
                    st.markdown(f"- {s}")

            with col2:
                st.markdown("### 🎯 Action Plan")
                st.markdown(
                    f'<div class="strategy-box">'
                    f'<b>Action:</b> {action}<br><br>'
                    f'<b>Timeline:</b> {timeline}<br><br>'
                    f'<b>Offer:</b> {offer}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── SHAP Explanation ──────────────────────────────
            st.markdown("### 🔍 SHAP Feature Explanation")
            st.info(
                "Positive SHAP = Pushes toward CHURN | "
                "Negative SHAP = Pushes toward STAY"
            )

            with st.spinner("Calculating SHAP..."):
                shap_df = get_shap_explanation(
                    explainer,
                    X_scaled,
                    feature_names
                )

            if shap_df is not None:
                colors_shap = [
                    '#FF4444' if v > 0 else '#00C851'
                    for v in shap_df['SHAP_Value']
                ]
                fig_shap = go.Figure(go.Bar(
                    x    = shap_df['SHAP_Value'],
                    y    = shap_df['Feature'],
                    orientation  = 'h',
                    marker_color = colors_shap
                ))
                fig_shap.update_layout(
                    height      = 400,
                    title       = "Top 10 Factors for This Customer",
                    xaxis_title = "SHAP Value"
                )
                st.plotly_chart(
                    fig_shap,
                    use_container_width=True
                )

                st.dataframe(
                    shap_df[[
                        'Feature', 'SHAP_Value'
                    ]].rename(columns={
                        'SHAP_Value': 'Impact'
                    }),
                    use_container_width=True
                )

    # ══════════════════════════════════════════════════════════
    # PAGE 3: BULK CSV PREDICTION
    # ══════════════════════════════════════════════════════════
    elif page == "📁 Bulk CSV Prediction":

        st.markdown("## 📁 Bulk Customer Prediction")
        st.info(
            "Upload a CSV file with customer data. "
            "The CSV should have the same columns as "
            "the cleaned telecom dataset."
        )

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload cleaned telecom dataset CSV"
        )

        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)

            st.success(
                f"✅ File uploaded! "
                f"Shape: {df_upload.shape}"
            )
            st.dataframe(
                df_upload.head(5),
                use_container_width=True
            )

            # Remove churn column if exists
            if 'Churn' in df_upload.columns:
                actual_churn = df_upload['Churn'].copy()
                df_upload    = df_upload.drop(
                    columns=['Churn']
                )
                has_actual = True
            else:
                has_actual     = False
                actual_churn   = None

            if st.button(
                "🔮 Predict for All Customers",
                use_container_width=True
            ):
                with st.spinner(
                    "Predicting for all customers..."
                ):
                    probs, preds, _ = predict_churn(
                        df_upload, model,
                        scaler, feature_names
                    )

                # Risk categories
                risks = [
                    get_risk_category(p)[0]
                    for p in probs
                ]

                # Results dataframe
                df_results = df_upload.copy()
                df_results['Churn_Probability'] = probs
                df_results['Churn_Probability_%'] = (
                    probs * 100
                ).round(2)
                df_results['Risk_Category'] = risks
                df_results['Predicted_Churn'] = preds

                if has_actual:
                    df_results['Actual_Churn'] = (
                        actual_churn.values
                    )

                # ── Summary ───────────────────────────────────
                st.markdown("---")
                st.markdown("## 📊 Prediction Summary")

                red_c    = (np.array(risks) == 'RED').sum()
                yellow_c = (np.array(risks) == 'YELLOW').sum()
                green_c  = (np.array(risks) == 'GREEN').sum()
                total    = len(risks)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Customers",
                        f"{total:,}"
                    )
                with col2:
                    st.metric(
                        "🔴 RED (Urgent)",
                        f"{red_c:,}",
                        f"{red_c/total*100:.1f}%"
                    )
                with col3:
                    st.metric(
                        "🟡 YELLOW (Monitor)",
                        f"{yellow_c:,}",
                        f"{yellow_c/total*100:.1f}%"
                    )
                with col4:
                    st.metric(
                        "🟢 GREEN (Safe)",
                        f"{green_c:,}",
                        f"{green_c/total*100:.1f}%"
                    )

                # ── Pie Chart ─────────────────────────────────
                col1, col2 = st.columns(2)

                with col1:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels = ['🔴 RED',
                                  '🟡 YELLOW',
                                  '🟢 GREEN'],
                        values = [red_c,
                                  yellow_c,
                                  green_c],
                        hole   = 0.4,
                        marker = dict(colors=[
                            '#FF4444',
                            '#FFD700',
                            '#00C851'
                        ])
                    )])
                    fig_pie.update_layout(
                        height = 350,
                        title  = "Risk Distribution"
                    )
                    st.plotly_chart(
                        fig_pie,
                        use_container_width=True
                    )

                with col2:
                    fig_bar = go.Figure(data=[
                        go.Bar(
                            x = ['🔴 RED',
                                 '🟡 YELLOW',
                                 '🟢 GREEN'],
                            y = [red_c,
                                 yellow_c,
                                 green_c],
                            marker_color = [
                                '#FF4444',
                                '#FFD700',
                                '#00C851'
                            ],
                            text = [
                                f'{red_c:,}',
                                f'{yellow_c:,}',
                                f'{green_c:,}'
                            ],
                            textposition = 'outside'
                        )
                    ])
                    fig_bar.update_layout(
                        height = 350,
                        title  = "Risk Count"
                    )
                    st.plotly_chart(
                        fig_bar,
                        use_container_width=True
                    )

                # ── Top 10 High Risk ──────────────────────────
                st.markdown(
                    "### 🔴 Top 10 Highest Risk Customers"
                )
                top10 = df_results.nlargest(
                    10, 'Churn_Probability'
                )[[
                    'Churn_Probability_%',
                    'Risk_Category',
                    'Predicted_Churn'
                ]]
                st.dataframe(
                    top10,
                    use_container_width=True
                )

                # ── Full Results ──────────────────────────────
                st.markdown("### 📋 Full Results")
                st.dataframe(
                    df_results[[
                        'Churn_Probability_%',
                        'Risk_Category',
                        'Predicted_Churn'
                    ]].head(100),
                    use_container_width=True
                )

                # ── Download ──────────────────────────────────
                st.markdown("### 💾 Download Results")
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label     = "⬇️ Download Full Results CSV",
                    data      = csv,
                    file_name = "churn_predictions.csv",
                    mime      = "text/csv",
                    use_container_width=True
                )

    # ══════════════════════════════════════════════════════════
    # PAGE 4: MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════
    elif page == "📊 Model Performance":

        st.markdown("## 📊 Model Performance")

        # Model results table
        st.markdown("### 📋 All Models Comparison")
        st.dataframe(
            model_results.style.highlight_max(
                subset=[
                    'Accuracy', 'Precision',
                    'Recall', 'F1-Score', 'AUC-ROC'
                ],
                color='lightgreen'
            ),
            use_container_width=True
        )

        st.markdown("---")

        # Bar chart comparison
        st.markdown("### 📊 Visual Comparison")

        metrics   = [
            'Accuracy', 'Precision',
            'Recall', 'F1-Score', 'AUC-ROC'
        ]
        fig_comp  = go.Figure()

        colors_models = [
            '#2E86AB', '#F18F01',
            '#3BB273', '#C73E1D', '#A23B72'
        ]

        for i, row in model_results.iterrows():
            try:
                vals = [
                    float(row[m])
                    for m in metrics
                ]
                fig_comp.add_trace(go.Bar(
                    name         = row['Model'],
                    x            = metrics,
                    y            = vals,
                    marker_color = colors_models[
                        i % len(colors_models)
                    ]
                ))
            except:
                pass

        fig_comp.update_layout(
            height      = 450,
            barmode     = 'group',
            title       = "All Models - All Metrics",
            yaxis_title = "Score",
            yaxis       = dict(range=[0, 1.1])
        )
        st.plotly_chart(
            fig_comp, use_container_width=True
        )

        st.markdown("---")

        # Project summary
        st.markdown("### 📋 Project Summary")
        if project_summary:
            col1, col2 = st.columns(2)
            items = list(project_summary.items())
            mid   = len(items) // 2

            with col1:
                for k, v in items[:mid]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<b>{k}</b>: {v}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            with col2:
                for k, v in items[mid:]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<b>{k}</b>: {v}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Risk segmentation info
        st.markdown("---")
        st.markdown("### 🎯 Risk Segmentation Logic")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                '<div class="red-card">'
                '🔴 RED<br>'
                'Probability ≥ 70%<br>'
                'IMMEDIATE ACTION<br>'
                '93.9% actual churn'
                '</div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                '<div class="yellow-card">'
                '🟡 YELLOW<br>'
                'Probability 40-70%<br>'
                'MONITOR CLOSELY<br>'
                '61.8% actual churn'
                '</div>',
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                '<div class="green-card">'
                '🟢 GREEN<br>'
                'Probability < 40%<br>'
                'SAFE CUSTOMER<br>'
                '1.4% actual churn'
                '</div>',
                unsafe_allow_html=True
            )

# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()