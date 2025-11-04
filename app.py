import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from openai import OpenAI
import os

# -----------------------------------------------------
# Streamlit Page Config (must be first command)
# -----------------------------------------------------
st.set_page_config(page_title="Daily Activity Classifier", layout="wide")

# -----------------------------------------------------
# OpenAI Client
# -----------------------------------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------------------------------
# App Title
# -----------------------------------------------------
st.title("🏋️‍♂️ Daily Activity Classifier")
st.markdown("""
Predict whether your day was **Active** or **Inactive** based on your metrics.  
You can enter data manually for a single day or upload a CSV for multiple days.
""")

# -----------------------------------------------------
# Load trained model and features
# -----------------------------------------------------
model = joblib.load("models/rf_model.pkl")
features = joblib.load("models/rf_features.pkl")

# -----------------------------------------------------
# Utility Functions
# -----------------------------------------------------
def lbs_to_kg(pounds):
    return pounds * 0.453592

def inches_to_cm(inches):
    return inches * 2.54

def calculate_bmr(sex, weight_kg, height_cm):
    """Mifflin-St Jeor Equation (assuming age 30)."""
    if sex.lower() == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * 30 + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * 30 - 161

# -----------------------------------------------------
# Prediction Section
# -----------------------------------------------------
st.header("1️⃣ Enter Your Metrics")
input_option = st.radio("Input type:", ["Single Day Manual Input", "Upload CSV"])

if input_option == "Single Day Manual Input":
    sex = st.selectbox("Sex", ["Male", "Female"])
    weight_lb = st.number_input("Weight (lbs)", min_value=0.0, value=160.0)
    height_in = st.number_input("Height (inches)", min_value=0.0, value=68.0)
    workout_minutes = st.number_input("Workout Minutes", min_value=0.0, value=30.0)
    calories_burned_input = st.number_input("Calories Burned During Workout", min_value=0.0, value=250.0)

    if st.button("Predict Single Day"):
        weight_kg = lbs_to_kg(weight_lb)
        height_cm = inches_to_cm(height_in)
        bmr = calculate_bmr(sex, weight_kg, height_cm)
        total_calories = calories_burned_input + bmr

        X_input = pd.DataFrame({
            "workout_minutes": [workout_minutes],
            "total_calories": [total_calories]
        })

        # Ensure feature alignment
        for col in features:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[features]

        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]

        st.subheader("Prediction")
        if prediction == 1:
            st.success(f"Active Day ✅ (Confidence: {round(prob[1]*100,2)}%)")
        else:
            st.warning(f"Inactive Day 💤 (Confidence: {round(prob[0]*100,2)}%)")

        st.subheader("Visualization")
        fig = px.bar(
            x=["Workout Minutes", "Total Calories"],
            y=[workout_minutes, total_calories],
            labels={"x": "Metric", "y": "Value"},
            title="Daily Metrics"
        )
        st.plotly_chart(fig)

        st.session_state["last_prediction"] = {
            "date": "Day 1",
            "active": bool(prediction),
            "calories": total_calories
        }

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Raw Data Preview", df.head())

        required_cols = ["sex", "weight_lb", "height_in", "workout_minutes", "calories_burned"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
        else:
            df["weight_kg"] = df["weight_lb"].apply(lbs_to_kg)
            df["height_cm"] = df["height_in"].apply(inches_to_cm)
            df["bmr"] = df.apply(lambda x: calculate_bmr(x["sex"], x["weight_kg"], x["height_cm"]), axis=1)
            df["total_calories"] = df["calories_burned"] + df["bmr"]

            X_input = pd.DataFrame({
                "workout_minutes": df["workout_minutes"],
                "total_calories": df["total_calories"]
            })
            for col in features:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[features]

            df["prediction"] = model.predict(X_input)
            df["prob_active"] = model.predict_proba(X_input)[:, 1]

            st.subheader("Predictions")
            st.write(df[["workout_minutes", "total_calories", "prediction", "prob_active"]])

            st.subheader("Visualization")
            x_vals = df["date"] if "date" in df.columns else [f"Day {i+1}" for i in range(len(df))]
            fig = px.bar(df, x=x_vals, y="prediction",
                         labels={"prediction": "Active (1) / Inactive (0)", "x": "Day"},
                         title="Activity Classification per Day")
            st.plotly_chart(fig)

# -----------------------------------------------------
# Ask Coach AI — OpenAI Cloud Model
# -----------------------------------------------------
st.header("2️⃣ Ask Coach AI")
user_q = st.text_area("Ask anything about improving your activity, routines, or habits:")

if st.button("Get Advice") and user_q.strip():
    last_prediction = st.session_state.get("last_prediction", None)
    context = ""
    if last_prediction:
        context = (
            f"On {last_prediction['date']}, you were classified as "
            f"{'Active ✅' if last_prediction['active'] else 'Inactive 💤'}, "
            f"with {round(last_prediction['calories'],1)} total calories burned.\n"
        )

    SAFETY_PROMPT = (
        "You are a motivational, safe, and practical health coach.\n"
        "Avoid medical advice or diagnoses.\n"
        "Focus on motivation, recovery, and consistency.\n"
        "Answer in short bullet points.\n"
    )
    full_prompt = SAFETY_PROMPT + context + "User: " + user_q

    with st.spinner("Thinking..."):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7,
                max_tokens=400
            )
            coach_answer = response.choices[0].message.content
            st.markdown(coach_answer)
        except Exception as e:
            st.error(f"⚠️ Error communicating with OpenAI: {e}")