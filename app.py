import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from openai import OpenAI

# -------------------------
# Initialize OpenAI client
# -------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# App config and title
# -------------------------
st.set_page_config(page_title="Daily Activity Classifier", layout="wide")
st.title("Daily Activity Classifier")
st.markdown("""
This app predicts whether a day is **active** or **inactive** based on activity metrics.
Enter your data manually for a single day or upload a CSV for multiple days.
""")

# -------------------------
# Load model and features
# -------------------------
model = joblib.load("models/rf_model.pkl")
features = joblib.load("models/rf_features.pkl")

# -------------------------
# Mapping UI-friendly names to model features
# -------------------------
FEATURE_MAP = {
    "Workout Minutes": "very_active_minutes",  # replace with actual training feature
    "Total Calories": "total_calories",
}

# -------------------------
# Helper: BMR calculation
# -------------------------
def calculate_bmr(sex, weight_kg, height_cm):
    """Mifflin-St Jeor Equation (age assumed 30)"""
    if sex.lower() == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * 30 + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * 30 - 161

# -------------------------
# User input section
# -------------------------
st.header("1️⃣ Enter Your Metrics")
input_option = st.radio("Input type:", ["Single Day Manual Input", "Upload CSV"])

if input_option == "Single Day Manual Input":
    sex = st.selectbox("Sex", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200)
    height = st.number_input("Height (cm)", min_value=100, max_value=250)
    workout_minutes = st.number_input("Workout Minutes", min_value=0)
    calories_burned_input = st.number_input("Calories Burned During Workout", min_value=0)

    if st.button("Predict Single Day"):
        bmr = calculate_bmr(sex, weight, height)
        total_calories = calories_burned_input + bmr

        # Build input DataFrame using FEATURE_MAP
        X_input = pd.DataFrame({
            FEATURE_MAP["Workout Minutes"]: [workout_minutes],
            FEATURE_MAP["Total Calories"]: [total_calories],
        })
        X_input = X_input[features]  # ensure column names match training

        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]

        st.subheader("Prediction")
        if prediction == 1:
            st.success(f"Active Day ✅ (Confidence: {round(prob[1]*100,2)}%)")
        else:
            st.warning(f"Inactive Day 💤 (Confidence: {round(prob[0]*100,2)}%)")

        st.subheader("Visualization")
        fig = px.bar(x=["Workout Minutes", "Total Calories"], 
                     y=[workout_minutes, total_calories],
                     labels={"x": "Metric", "y": "Value"},
                     title="Daily Metrics")
        st.plotly_chart(fig)

        st.session_state["last_prediction"] = {
            "date": "Day 1",
            "active": bool(prediction),
            "calories": total_calories
        }

else:  # CSV upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Raw Data", df.head())

        required_cols = ["sex", "weight_kg", "height_cm", "workout_minutes", "calories_burned"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
        else:
            df["bmr"] = df.apply(lambda x: calculate_bmr(x["sex"], x["weight_kg"], x["height_cm"]), axis=1)
            df["total_calories"] = df["calories_burned"] + df["bmr"]

            # Map UI-friendly CSV column to model feature names
            X_input = pd.DataFrame({
                FEATURE_MAP["Workout Minutes"]: df["workout_minutes"],
                FEATURE_MAP["Total Calories"]: df["total_calories"]
            })
            X_input = X_input[features]

            df["prediction"] = model.predict(X_input)
            df["prob_active"] = model.predict_proba(X_input)[:,1]

            st.subheader("Predictions")
            st.write(df[["workout_minutes", "total_calories", "prediction", "prob_active"]])

            st.subheader("Visualization")
            x_vals = df.index if "date" not in df.columns else df["date"]
            fig = px.bar(df, x=x_vals, y="prediction", labels={"prediction":"Active (1) / Inactive (0)", "x":"Day"}, title="Activity Classification per Day")
            st.plotly_chart(fig)

            last_row = df.iloc[-1]
            st.session_state["last_prediction"] = {
                "date": str(last_row.get("date", "Last Day")),
                "active": bool(last_row["prediction"]),
                "calories": last_row["total_calories"]
            }

# -------------------------
# Ask Coach AI Section
# -------------------------
st.header("2️⃣ Ask Coach AI")
user_q = st.text_area("Ask anything about improving your activity, routines, or habits:")

last_prediction = st.session_state.get("last_prediction", None)
context = ""
if last_prediction:
    context = (
        f"On {last_prediction['date']}, you were classified as "
        f"{'Active ✅' if last_prediction['active'] else 'Inactive 💤'}, "
        f"with {last_prediction['calories']} total calories burned.\n"
    )

SAFETY_PROMPT = (
    "You are an activity and recovery coach.\n"
    "Do not provide medical advice or diagnoses.\n"
    "Give practical, safe, data-informed suggestions only.\n"
    "Answer in bullet points."
)

if st.button("Get Advice") and user_q.strip():
    full_prompt = SAFETY_PROMPT + "\n" + context + "User: " + user_q
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.7
    )
    coach_answer = response.choices[0].message.content
    st.markdown(coach_answer)

