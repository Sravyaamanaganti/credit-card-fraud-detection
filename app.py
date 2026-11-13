
# Step 9.2.1: Load our saved Random Forest model

import pandas as pd
import streamlit as st
import joblib

# Load the trained model
rf_model = joblib.load("models/random_forest_model.pkl")

st.title("Credit Card Fraud Detection")

st.success("Random Forest model loaded successfully!")

# Step 9.2.3: Get transaction details from the user

st.subheader("Enter Transaction Details")

amount = st.number_input(
    "Transaction Amount ($)",
    min_value=0.0,
    value=100.0
)

hour = st.number_input(
    "Transaction Hour (0-23)",
    min_value=0,
    max_value=23,
    value=12
)

category = st.selectbox(
    "Transaction Category",
    ["Grocery", "Electronics", "Shopping",
     "Rent", "Movie", "Travel", "Jewellery"]
)

# Step 9.2.4: Predict Fraud

if st.button("Predict Fraud"):

    transaction = pd.DataFrame(
        [{name: 0 for name in rf_model.feature_names_in_}]
    )

    transaction["amount"] = amount
    transaction["hour"] = hour
    transaction[f"category_{category}"] = 1

    fraud_probability = rf_model.predict_proba(transaction)[0][1]

    prediction = int(fraud_probability >= 0.30)

    if prediction == 1:
        st.warning("Prediction: Potential Fraud (1)")
    else:
        st.success("Prediction: Genuine (0)")

    st.write(f"Model fraud score: {fraud_probability:.2%}")
