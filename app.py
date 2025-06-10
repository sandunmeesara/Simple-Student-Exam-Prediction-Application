import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("model/nn_model.h5")
scaler = joblib.load("model/scaler.pkl")

st.title("🎓 Student Exam Pass Predictor")

# Input fields
hours = st.number_input("Hours Studied", 0.0, 20.0, 10.0)
attendance = st.slider("Attendance (%)", 50, 100, 85)
sleep = st.slider("Sleep Hours", 4.0, 10.0, 7.0)
score = st.slider("Previous Score", 30, 100, 65)
tutor = st.radio("Used Tutor?", ["Yes", "No"])

# Convert inputs
input_data = np.array([[hours, attendance, sleep, score, 1 if tutor == "Yes" else 0]])
input_scaled = scaler.transform(input_data)

# Predict
prob = model.predict(input_scaled)[0][0]
st.write(f"**Pass Probability:** {prob:.2f}")
st.success("🎉 PASS Prediction ✅") if prob > 0.5 else st.error("❌ FAIL Prediction")
