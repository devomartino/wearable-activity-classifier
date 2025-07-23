import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/rf_model.pkl')

st.title("Fitbit Activity Classifier")
st.markdown("Enter your activity metrics below to predict if the day is Active or Inactive.")

# User input fields
very_active = st.number_input("Very Active Minutes", min_value=0)
fairly_active = st.number_input("Fairly Active Minutes", min_value=0)
lightly_active = st.number_input("Lightly Active Minutes", min_value=0)
sedentary = st.number_input("Sedentary Minutes", min_value=0)
calories = st.number_input("Calories Burned", min_value=0)

# Predict when the button is clicked
if st.button("Predict"):
    input_data = np.array([[sedentary, calories, lightly_active, fairly_active, very_active]])
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"Prediction: Active Day ✅ (Confidence: {round(prob[1]*100, 2)}%)")
    else:
        st.warning(f"Prediction: Inactive Day 💤 (Confidence: {round(prob[0]*100, 2)}%)")