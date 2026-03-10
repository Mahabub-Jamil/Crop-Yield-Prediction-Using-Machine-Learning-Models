import streamlit as st
import numpy as np
import joblib

# Load model and encoders
model = joblib.load('D:/AI Project/Capstone Project/Final/best_crop_model.pkl')
le_crop = joblib.load('D:/AI Project/Capstone Project/Final/label_encoder_crop.pkl')
le_soil = joblib.load('D:/AI Project/Capstone Project/Final/label_encoder_soil.pkl')

soil_labels = le_soil.classes_

st.set_page_config(page_title="Crop Predictor", layout="centered")
st.title("🌾 Crop Prediction System")
st.markdown("Enter the following parameters to predict the most suitable crop.")

# Input fields
nitrogen = st.number_input("Nitrogen", min_value=0.0, format="%.2f")
phosphorus = st.number_input("Phosphorus", min_value=0.0, format="%.2f")
potassium = st.number_input("Potassium", min_value=0.0, format="%.2f")
temperature = st.number_input("Temperature (°C)", format="%.2f")
humidity = st.number_input("Humidity (%)", format="%.2f")
ph_value = st.number_input("pH Value", format="%.2f")
rainfall = st.number_input("Rainfall (mm)", format="%.2f")
moisture = st.number_input("Moisture (%)", format="%.2f")

soil_type = st.selectbox("Soil Type", soil_labels)

# Predict button
if st.button("Predict Crop"):
    try:
        soil_encoded = le_soil.transform([soil_type])[0]
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity,
                              ph_value, rainfall, moisture, soil_encoded]])
        prediction = model.predict(features)
        crop_predicted = le_crop.inverse_transform(prediction)[0]
        st.success(f"✅ Recommended Crop: **{crop_predicted}**")
    except Exception as e:
        st.error(f"Error: {e}")
