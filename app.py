import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import time
import plotly.graph_objects as go


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)




st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Animated gradient background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1e3c72);
    background-size: 400% 400%;
    animation: gradientShift 18s ease infinite;
}

@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Fade + slide-in for the whole main block */
[data-testid="stAppViewContainer"] > .main {
    animation: fadeSlideIn 0.8s ease-out;
}
@keyframes fadeSlideIn {
    from {opacity: 0; transform: translateY(18px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Title styling */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 3s linear infinite;
    margin-bottom: 0.2rem;
}
@keyframes shine {
    to { background-position: 200% center; }
}
.hero-subtitle {
    text-align: center;
    color: rgba(255,255,255,0.75);
    font-weight: 300;
    margin-bottom: 1.8rem;
}

/* Glass card container */
.glass-card {
    background: rgba(255, 255, 255, 0.07);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeSlideIn 0.9s ease-out;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
}

/* Section labels inside cards */
.section-label {
    font-size: 1.05rem;
    font-weight: 600;
    color: #7fd8ff;
    margin-bottom: 0.6rem;
    letter-spacing: 0.3px;
}

/* Predict button */
div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #3a7bd5, #00d2ff);
    color: white;
    font-weight: 600;
    font-size: 1.05rem;
    padding: 0.7rem 0;
    border: none;
    border-radius: 12px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 4px 18px rgba(0, 210, 255, 0.35);
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 26px rgba(0, 210, 255, 0.55);
}
div.stButton > button:active {
    transform: translateY(0) scale(0.99);
}

/* Result banners */
.result-banner {
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    margin-top: 0.8rem;
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(0.85); }
    100% { opacity: 1; transform: scale(1); }
}
.banner-churn {
    background: rgba(255, 70, 70, 0.15);
    border: 1px solid rgba(255, 70, 70, 0.4);
    color: #ff8a8a;
}
.banner-safe {
    background: rgba(60, 220, 130, 0.15);
    border: 1px solid rgba(60, 220, 130, 0.4);
    color: #7bf0b0;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] {
    background: rgba(10, 15, 25, 0.55);
    backdrop-filter: blur(10px);
}

hr {
    border-color: rgba(255,255,255,0.15);
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL & PREPROCESSORS (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('onehot_encoder.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-title">🔮 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered insight into whether a customer is likely to stay or leave</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR — inputs live here so the main area stays clean for results
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧾 Customer Profile")
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 100, 30)
    tenure = st.slider("📅 Tenure (years)", 0, 10, 5)

    st.markdown("### 💳 Account Details")
    credit_score = st.number_input("Credit Score", value=600)
    balance = st.number_input("Balance ($)", value=0.0)
    estimated_salary = st.number_input("Estimated Salary ($)", value=50000.0)
    num_of_products = st.slider("Number of Products", 1, 4, 1)

    st.markdown("### ⚙️ Engagement")
    has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")
    is_active_member = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown("---")
    predict_clicked = st.button("🚀 Predict Churn Risk")

# ----------------------------------------------------------------------------
# MAIN AREA — summary card + prediction
# ----------------------------------------------------------------------------
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Profile Summary</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame({
        "Feature": ["Geography", "Gender", "Age", "Tenure", "Credit Score",
                    "Balance", "Salary", "Products", "Credit Card", "Active Member"],
        "Value": [geography, gender, age, tenure, credit_score,
                  f"${balance:,.2f}", f"${estimated_salary:,.2f}", num_of_products,
                  "Yes" if has_cr_card else "No", "Yes" if is_active_member else "No"],
    })
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Prediction Result</div>', unsafe_allow_html=True)

    if predict_clicked:
        with st.spinner("Running the model..."):
            time.sleep(0.6)  # small delay so the spinner is visible

            # 1. Base input DataFrame
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [label_encoder_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_of_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active_member],
                'EstimatedSalary': [estimated_salary],
            })

            # 2. One-hot encode Geography
            geo_input = pd.DataFrame([[geography]], columns=['Geography'])
            geography_encoded = onehot_encoder_geo.transform(geo_input)
            if hasattr(geography_encoded, "toarray"):
                geography_encoded = geography_encoded.toarray()
            geography_df = pd.DataFrame(
                geography_encoded,
                columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
            )

            # 3. Combine
            input_data = pd.concat(
                [input_data.reset_index(drop=True), geography_df.reset_index(drop=True)],
                axis=1
            )

            # 4. Reorder columns to match scaler's training order
            if hasattr(scaler, "feature_names_in_"):
                input_data = input_data[scaler.feature_names_in_]

            # 5. Scale
            scaled_input = scaler.transform(input_data)

            # 6. Predict
            model_prediction = model.predict(scaled_input, verbose=0)
            prediction_probability = float(model_prediction[0][0])

        # --- Animated gauge chart ---
        gauge_color = "#ff6b6b" if prediction_probability > 0.5 else "#3ddc97"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_probability * 100,
            number={'suffix': "%", 'font': {'size': 42, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': gauge_color},
                'bgcolor': 'rgba(255,255,255,0.05)',
                'borderwidth': 1,
                'bordercolor': 'rgba(255,255,255,0.2)',
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(61,220,151,0.15)'},
                    {'range': [50, 100], 'color': 'rgba(255,107,107,0.15)'},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8,
                    'value': prediction_probability * 100,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            margin=dict(l=20, r=20, t=30, b=10),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

        if prediction_probability > 0.5:
            st.markdown(
                f'<div class="result-banner banner-churn">⚠️ High churn risk — '
                f'{prediction_probability:.1%} likelihood the customer leaves.</div>',
                unsafe_allow_html=True,
            )
            st.snow()
        else:
            st.markdown(
                f'<div class="result-banner banner-safe">✅ Low churn risk — '
                f'{prediction_probability:.1%} likelihood the customer leaves.</div>',
                unsafe_allow_html=True,
            )
            st.balloons()
    else:
        st.info("Fill in the customer profile on the left, then click **🚀 Predict Churn Risk**.")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p style="text-align:center; color:rgba(255,255,255,0.4); margin-top:2rem; font-size:0.85rem;">'
    'Built with Streamlit · TensorFlow · Plotly</p>',
    unsafe_allow_html=True,
)
