import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    color:#0E76FD;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:30px;
}

.block-container{
    padding-top:2rem;
}

.stButton>button{
    width:100%;
    background:#0E76FD;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
}

.stButton>button:hover{
    background:#0958c9;
}

.metric-card{
    padding:15px;
    border-radius:12px;
    background:white;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------

st.markdown(
    "<div class='title'>🤖 Customer Churn Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Artificial Neural Network (PyTorch)</div>",
    unsafe_allow_html=True
)

# -------------------------------
# Model Definition
# -------------------------------

class CustomerChurnANN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(11,16),
            nn.ReLU(),

            nn.Linear(16,8),
            nn.ReLU(),

            nn.Linear(8,1),
            nn.Sigmoid()

        )

    def forward(self,x):

        return self.network(x)

# -------------------------------
# Load Model
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "customer_churn_ann.pth"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = CustomerChurnANN()

model.load_state_dict(
    torch.load(MODEL_PATH,map_location="cpu")
)

model.eval()

scaler = joblib.load(SCALER_PATH)

# -------------------------------
# Input Section
# -------------------------------

st.subheader("📋 Customer Information")

col1,col2 = st.columns(2)

# -------------------------------
# Left Column
# -------------------------------

with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

# -------------------------------
# Right Column
# -------------------------------

with col2:

    num_products = st.selectbox(
        "Number of Products",
        [1,2,3,4]
    )

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    geography = st.selectbox(
        "Geography",
        ["France","Germany","Spain"]
    )

    has_credit_card = st.selectbox(
        "Has Credit Card",
        ["Yes","No"]
    )

    active_member = st.selectbox(
        "Is Active Member",
        ["Yes","No"]
    )

# -------------------------------
# Encode Inputs
# -------------------------------

gender = 1 if gender == "Male" else 0

has_credit_card = 1 if has_credit_card == "Yes" else 0

active_member = 1 if active_member == "Yes" else 0

geography_germany = 1 if geography == "Germany" else 0

geography_spain = 1 if geography == "Spain" else 0

# -------------------------------
# Predict Button
# -------------------------------

st.markdown("---")

predict = st.button("🚀 Predict Customer Churn")
# -------------------------------
# Prediction
# -------------------------------

if predict:

    input_data = np.array([[
        credit_score,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_credit_card,
        active_member,
        estimated_salary,
        geography_germany,
        geography_spain
    ]])

    # Scale Features
    input_scaled = scaler.transform(input_data)

    input_tensor = torch.tensor(
        input_scaled,
        dtype=torch.float32
    )

    # Predict
    with torch.no_grad():

        probability = model(input_tensor).item()

    prediction = "Churn" if probability >= 0.5 else "Stay"

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    # Probability
    st.metric(
        label="Churn Probability",
        value=f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    # Risk Level
    if probability < 0.30:

        st.success("🟢 Low Risk Customer")

    elif probability < 0.70:

        st.warning("🟡 Medium Risk Customer")

    else:

        st.error("🔴 High Risk Customer")

    # Final Prediction
    if prediction == "Stay":

        st.success("✅ Customer is likely to Stay")

    else:

        st.error("❌ Customer is likely to Churn")

    st.markdown("---")

    st.subheader("🤖 AI Recommendation")

    if probability < 0.30:

        st.info("""
Customer has a low probability of churn.

### Suggested Actions

- Continue regular engagement
- Maintain current service quality
- Reward loyalty with offers
        """)

    elif probability < 0.70:

        st.warning("""
Customer has a moderate chance of churn.

### Suggested Actions

- Offer personalized discounts
- Send promotional emails
- Improve customer support
        """)

    else:

        st.error("""
Customer is highly likely to churn.

### Suggested Actions

- Immediate customer call
- Provide special retention offer
- Assign relationship manager
- Offer premium membership benefits
        """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", "PyTorch ANN")

    with col2:
        st.metric("Task", "Binary Classification")

    with col3:
        st.metric("Classes", "Stay / Churn")

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.markdown(
"""
<div style="text-align:center;color:gray">

Developed by <b>Dipanshu Shukla</b>

Customer Churn Prediction using Artificial Neural Network (PyTorch)

</div>
""",
unsafe_allow_html=True
)