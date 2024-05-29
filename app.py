import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')

# Function to make predictions
def predict_crop_type(model, scaler, poly, temperature, moisture, humidity):
    # Transform and scale the input features
    input_data = np.array([[temperature, moisture, humidity]])
    input_data_poly = poly.transform(input_data)
    input_data_scaled = scaler.transform(input_data_poly)
    # Predict the crop type
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Streamlit UI
st.title('Crop Type Prediction')

# Input form
st.subheader('Enter environmental parameters:')
temperature = st.number_input('Temperature')
moisture = st.number_input('Moisture')
humidity = st.number_input('Humidity')

# Predict button
if st.button('Predict'):
    # Make prediction
    prediction = predict_crop_type(ensemble_model, scaler, poly, temperature, moisture, humidity)
    st.write(f'Predicted crop type: {prediction}')
