import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"
OUTPUT_DIR = SCRIPTS_DIR / "output"

# Streamlit Page Config
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .sidebar .sidebar-content {background-color: #2c3e50; color: white;}
    h1, h2, h3 {color: #2c3e50;}
    .stButton>button {background-color: #3498db; color: white; font-weight: bold; height: 3em; border-radius: 8px;}
    .stButton>button:hover {background-color: #2980b9;}
    </style>
""", unsafe_allow_html=True)

# Cache loading models to avoid reloading on every UI interaction
@st.cache_resource
def load_models():
    try:
        model = joblib.load(OUTPUT_DIR / "best_model.pkl")
        scaler = joblib.load(OUTPUT_DIR / "scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_models()

# Sidebar Navigation
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio("Go to:", ["📊 Overview & EDA", "⚙️ Model Performance", "🔍 Live Fraud Simulator"])

st.sidebar.markdown("---")
st.sidebar.info("Built with **Streamlit** and **Scikit-Learn**\n\nEnsemble Learning approach for detecting Credit Card Fraud. Ready for Resume & Portfolio.")

def load_image_safely(filename):
    """Utility to load images gracefully if they exist."""
    path = OUTPUT_DIR / filename
    if path.exists():
        return Image.open(path)
    return None

if page == "📊 Overview & EDA":
    st.title("📊 Credit Card Fraud Detection")
    st.subheader("Project Overview")
    st.markdown("""
        This project aims to detect fraudulent credit card transactions using **Ensemble Machine Learning** techniques.
        The dataset contains transactions made by European cardholders in September 2013 over two days.
        
        **Key Challenge:** Extreme Class Imbalance. Only around 0.172% of transactions are fraudulent!
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases", "492")
    col3.metric("Normal Cases", "284,315")
    
    st.markdown("---")
    st.subheader("Exploratory Data Analysis")
    st.markdown("Here are some key visualizations from our data exploration phase:")
    
    col_a, col_b = st.columns(2)
    dist_img = load_image_safely("class_distribution.png")
    if dist_img:
        col_a.image(dist_img, caption="Highly Imbalanced Class Distribution", use_container_width=True)
    else:
        col_a.warning("Class distribution image not found. Run `python scripts/main.py` first.")
        
    amount_img = load_image_safely("amount_analysis.png")
    if amount_img:
        col_b.image(amount_img, caption="Transaction Amount Distribution Overview", use_container_width=True)
        
    st.markdown("### Feature Analysis")
    corr_img = load_image_safely("correlation_analysis.png")
    if corr_img:
        st.image(corr_img, caption="Feature Correlation Heatmap", use_container_width=True)

elif page == "⚙️ Model Performance":
    st.title("⚙️ Model Architecture & Performance")
    st.markdown("""
        To combat extreme class imbalance, we applied **SMOTE (Synthetic Minority Over-sampling Technique)**.
        We evaluated multiple models (Logistic Regression, Decision Trees, Random Forests, Gradient Boosting)
        and combined the best ones into a **Voting Classifier ensemble**.
    """)
    
    st.subheader("Data Resampling Impact")
    resample_img = load_image_safely("resampling_comparison.png")
    if resample_img:
        st.image(resample_img, caption="Effect of SMOTE and Undersampling", use_container_width=True)
        
    st.markdown("---")
    st.subheader("Model Evaluation Metrics")
    col_a, col_b = st.columns(2)
    
    roc_img = load_image_safely("roc_curves.png")
    if roc_img:
        col_a.image(roc_img, caption="ROC-AUC Curves", use_container_width=True)
        
    conf_img = load_image_safely("confusion_matrices.png")
    if conf_img:
        col_b.image(conf_img, caption="Confusion Matrices", use_container_width=True)
        
    st.subheader("Feature Importance & Threshold Optimization")
    feat_img = load_image_safely("feature_importance.png")
    if feat_img:
        st.image(feat_img, caption="Random Forest Feature Importance", use_container_width=True)
        
    thresh_img = load_image_safely("threshold_tuning.png")
    if thresh_img:
        st.image(thresh_img, caption="Threshold Tuning to Optimize F1-Score", use_container_width=True)

elif page == "🔍 Live Fraud Simulator":
    st.title("🔍 Live Fraud Detection Simulator")
    st.markdown("Test the deployed ensemble model in real-time by tweaking transaction features.")
    
    if model is None or scaler is None:
        st.error("⚠️ Model or Scaler not found! Please run `python scripts/main.py` first to train and export the models.")
    else:
        st.success(f"✅ Prediction Engine Loaded: **{model.__class__.__name__}**")
        
        st.markdown("### Transaction Details")
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=150.0, step=10.0)
        with col2:
            time = st.number_input("Time (Seconds from start)", min_value=0.0, max_value=180000.0, value=3600.0, step=100.0)
            
        st.markdown("### Principal Components (V1 - V28)")
        st.info("These 28 features are the result of a PCA transformation applied to the original dataset by the card issuer for confidentiality.")
        
        # Create an array of 28 zeros
        v_features = np.zeros(28)
        
        # We will only expose sliders for the top 9 most important features (usually V14, V10, V4, etc. based on RF importance)
        # But for UI simplicity, let's expose V1 to V9
        v_cols_1, v_cols_2, v_cols_3 = st.columns(3)
        
        with v_cols_1:
            v_features[0] = st.slider("V1", -5.0, 5.0, -1.0)
            v_features[3] = st.slider("V4", -5.0, 5.0, 0.0)
            v_features[6] = st.slider("V7", -5.0, 5.0, 0.0)
        with v_cols_2:
            v_features[1] = st.slider("V2", -5.0, 5.0, 1.0)
            v_features[4] = st.slider("V5", -5.0, 5.0, 0.0)
            v_features[7] = st.slider("V8", -5.0, 5.0, 0.0)
        with v_cols_3:
            v_features[2] = st.slider("V3", -5.0, 5.0, 0.0)
            v_features[5] = st.slider("V6", -5.0, 5.0, 0.0)
            v_features[8] = st.slider("V9", -5.0, 5.0, 0.0)
            
        # Give option to inject a highly suspicious pattern
        if st.checkbox("💉 Inject Suspicious Fraud Pattern (demo)"):
            # Known fraud patterns often have extreme negative V14, V10, V12, and positive V4, V11
            v_features[13] = -8.0 # V14
            v_features[9] = -5.0  # V10
            v_features[3] = 4.0   # V4
            amount = 999.99
            
        if st.button("Predict Transaction Status", use_container_width=True):
            with st.spinner("Analyzing transaction patterns..."):
                # Scale Amount and Time
                # The scaler expects a 2D array: [[Amount, Time]]
                scaled_inputs = scaler.transform([[amount, time]])
                amount_scaled, time_scaled = scaled_inputs[0][0], scaled_inputs[0][1]
                
                # Assemble final feature vector: [V1, ..., V28, Amount_scaled, Time_scaled]
                feature_vector = list(v_features) + [amount_scaled, time_scaled]
                feature_array = np.array(feature_vector).reshape(1, -1)
                
                # Make Prediction
                prediction = model.predict(feature_array)[0]
                
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(feature_array)[0][1]
                    st.progress(float(prob))
                    prob_text = f"Confidence: {prob * 100:.2f}%"
                else:
                    prob_text = ""
                
                if prediction == 1:
                    st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED!** {prob_text}")
                else:
                    st.success(f"✅ **Transaction Approved (Legitimate)** {prob_text}")
