import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Load the trained model
model = joblib.load('models/rf_model.pkl')

st.set_page_config(page_title="Activity Classifier", layout="wide")

st.title("Activity Classifier")

st.markdown("""
This app predicts whether a day is **active** or **inactive** based on activity tracker metrics.  
Upload or enter your metrics, and the model will provide predictions along with visualizations.
""")

# Sidebar for inputs
st.sidebar.header("Select Input Method")
input_method = st.sidebar.radio("Choose one:", ["Single Day Input", "Upload CSV"])

show_prob = st.sidebar.checkbox("Show Prediction Confidence", value=True)

# Function to predict
def predict_day(df):
    input_data = df[['Sedentary', 'Calories', 'Lightly Active', 'Fairly Active', 'Very Active']].values
    predictions = model.predict(input_data)
    probs = model.predict_proba(input_data)
    df['Prediction'] = ['Active ✅' if p==1 else 'Inactive 💤' for p in predictions]
    if show_prob:
        df['Confidence'] = [round(prob[1]*100,2) if p==1 else round(prob[0]*100,2) for p, prob in zip(predictions, probs)]
    return df

# Single day input
if input_method == "Single Day Input":
    st.sidebar.header("Enter Daily Metrics")
    very_active = st.sidebar.number_input("Very Active Minutes", min_value=0)
    fairly_active = st.sidebar.number_input("Fairly Active Minutes", min_value=0)
    lightly_active = st.sidebar.number_input("Lightly Active Minutes", min_value=0)
    sedentary = st.sidebar.number_input("Sedentary Minutes", min_value=0)
    calories = st.sidebar.number_input("Calories Burned", min_value=0)

    if st.button("Predict Single Day"):
        try:
            df_input = pd.DataFrame({
                "Sedentary": [sedentary],
                "Calories": [calories],
                "Lightly Active": [lightly_active],
                "Fairly Active": [fairly_active],
                "Very Active": [very_active]
            })
            result_df = predict_day(df_input)

            tab1, tab2 = st.tabs(["Prediction", "Visualization"])
            with tab1:
                st.subheader("Prediction Results")
                st.dataframe(result_df)
                st.download_button("Download Prediction", result_df.to_csv(index=False), "prediction.csv")
            with tab2:
                st.subheader("Activity Visualization")
                fig = px.bar(df_input.T.reset_index(), x='index', y=0,
                             color='index', text=0)
                fig.update_layout(showlegend=False, xaxis_title="Metric", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# CSV upload
elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = ['Sedentary', 'Calories', 'Lightly Active', 'Fairly Active', 'Very Active']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {', '.join(required_cols)}")
            else:
                result_df = predict_day(df)
                tab1, tab2 = st.tabs(["Predictions", "Visualization"])
                with tab1:
                    st.subheader("Prediction Results")
                    st.dataframe(result_df)
                    st.download_button("Download Predictions", result_df.to_csv(index=False), "predictions.csv")
                with tab2:
                    st.subheader("Activity Visualization")
                    fig = px.bar(result_df.melt(id_vars=['Prediction'], value_vars=required_cols),
                                 x='variable', y='value', color='variable', text='value')
                    fig.update_layout(showlegend=False, xaxis_title="Metric", yaxis_title="Value")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing CSV: {e}")