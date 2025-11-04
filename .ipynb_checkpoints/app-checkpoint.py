import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from openai import OpenAI

# -------------------------
# Initialize OpenAI client safely
# -------------------------
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    llm_enabled = True
except Exception:
    client = None
    llm_enabled = False
    st.warning("⚠️ OpenAI API key not found — Coach AI disabled.")

# -------------------------
# App config and title
# -------------------------
st.set_page_config(page_title="Daily Activity Classifier", layout="wide")
st.title("🏋️‍♂️ Daily Activity Classifier")
st.markdown("""
Predict whether your day was **Active** or **Inactive** based on your metrics.  
Enter your data manually or upload a CSV for multiple days.
""")

# -------------------------
# Load model and features
# -------------------------
model = joblib.load("models/rf_model.pkl")
features = joblib.load("models/rf_features.pkl")

# -------------------------
# Feature mapping
# -------------------------
FEATURE_MAP = {
    "Workout Minutes": "Workout_Minutes",
    "Total Calories": "Total_Calories",
}

# -------------------------
# Helper: BMR calculation
# -------------------------
def calculate_bmr(sex, weight_lbs, height_in):
    """Mifflin-St Jeor Equation (age assumed 30)"""
    weight_kg = weight_lbs * 0.453592
    height_cm = height_in * 2.54
    if sex.lower() == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * 30 + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * 30 - 161

# -------------------------
# Input section
# -------------------------
st.header("1️⃣ Enter Your Metrics")
input_option = st.radio("Input type:", ["Single Day Manual Input", "Upload CSV"])

if input_option == "Single Day Manual Input":
    sex = st.selectbox("Sex", ["Male", "Female"])
    weight_lbs = st.number_input("Weight (lbs)", min_value=50.0, step=1.0)
    height_in = st.number_input("Height (inches)", min_value=48.0, step=0.5)
    workout_minutes = st.number_input("Workout Minutes", min_value=0.0, step=1.0)
    calories_burned_input = st.number_input("Calories Burned During Workout", min_value=0.0, step=10.0)

    if st.button("Predict Single Day"):
        bmr = calculate_bmr(sex, weight_lbs, height_in)
        total_calories = calories_burned_input + bmr

        # Build and align input DataFrame
        X_input = pd.DataFrame({
            "Workout_Minutes": [workout_minutes],
            "Total_Calories": [total_calories]
        })

        # Align to expected model features
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

else:  # CSV upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Raw Data Preview", df.head())

        required_cols = ["sex", "weight_lbs", "height_in", "workout_minutes", "calories_burned"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns in CSV: {missing}")
        else:
            df["bmr"] = df.apply(lambda x: calculate_bmr(x["sex"], x["weight_lbs"], x["height_in"]), axis=1)
            df["Total_Calories"] = df["calories_burned"] + df["bmr"]

            X_input = pd.DataFrame({
                "Workout_Minutes": df["workout_minutes"],
                "Total_Calories": df["Total_Calories"]
            })
            for col in features:
                if col not in X_input.columns:
                    X_input[col] = 0
            X_input = X_input[features]

            df["prediction"] = model.predict(X_input)
            df["prob_active"] = model.predict_proba(X_input)[:,1]

            st.subheader("Predictions")
            st.write(df[["workout_minutes", "Total_Calories", "prediction", "prob_active"]])

            st.subheader("Visualization")
            x_vals = df["date"] if "date" in df.columns else [f"Day {i+1}" for i in range(len(df))]
            fig = px.bar(df, x=x_vals, y="prediction",
                         labels={"prediction": "Active (1) / Inactive (0)", "x": "Day"},
                         title="Activity Classification per Day")
            st.plotly_chart(fig)

            last_row = df.iloc[-1]
            st.session_state["last_prediction"] = {
                "date": str(last_row.get("date", "Last Day")),
                "active": bool(last_row["prediction"]),
                "calories": last_row["Total_Calories"]
            }

# -------------------------
# Ask Coach AI
# -------------------------
st.header("2️⃣ Ask Coach AI")
user_q = st.text_area("Ask anything about improving your activity, routines, or habits:")

if llm_enabled and st.button("Get Advice") and user_q.strip():
    last_prediction = st.session_state.get("last_prediction", None)
    context = ""
    if last_prediction:
        context = (
            f"On {last_prediction['date']}, you were classified as "
            f"{'Active ✅' if last_prediction['active'] else 'Inactive 💤'}, "
            f"with {round(last_prediction['calories'],1)} total calories burned.\n"
        )

    SAFETY_PROMPT = (
        "You are an activity and recovery coach.\n"
        "Do not provide medical advice or diagnoses.\n"
        "Give practical, safe, and motivational suggestions only.\n"
        "Answer in bullet points."
    )

    full_prompt = SAFETY_PROMPT + "\n" + context + "User: " + user_q

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7
        )
        coach_answer = response.choices[0].message.content
        st.markdown(coach_answer)
    except Exception as e:
        st.error("Coach AI is temporarily unavailable. Try again later.")
elif not llm_enabled:
    st.info("Add your OpenAI API key in Streamlit Secrets to enable the Coach AI panel.")
