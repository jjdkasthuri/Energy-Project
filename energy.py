import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model_path = r'C:\Users\ASUS\Downloads\Energy Project\model.h5'
model = load_model(model_path)

# Initialize scaler with example parameters (mean and std)
# Replace these with actual parameters used during model training
mean = np.array([0.0] * 8)  # Example means for all 8 features
std = np.array([1.0] * 8)   # Example standard deviations for all 8 features

scaler = StandardScaler()
scaler.mean_ = mean
scaler.scale_ = std
scaler.n_features_in_ = len(mean)

# Set the title of the app
st.title("Energy Prediction")

# Input fields for energy amount and time step
energy_amount = st.number_input("Energy Amount", min_value=0.0, format="%.2f", step=0.01)
time_step = st.number_input("Time Step", min_value=0.0, format="%.2f", step=0.01)

# Input field for ground truth value
ground_truth = st.number_input("Ground Truth", min_value=0.0, format="%.2f", step=0.01)

# Predict button
if st.button("Predict"):
    try:
        # Prepare the input data for the model
        # Fill other features with default values (e.g., 0.0) or suitable values
        input_data = pd.DataFrame([[energy_amount, time_step, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 
                                  columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8'])
        
        # Standardize the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make a prediction
        prediction = model.predict(input_data_scaled)[0][0]  # Assuming the model outputs a single prediction
        
        # Display the results
        st.write(f"Ground Truth Value: {ground_truth:.2f}")
        st.write(f"Predicted Energy Value: {prediction:.2f}")
        
        # Check condition
        if ground_truth > prediction:
            st.write("Eligible for further consideration to become a prosumer.")
        else:
            st.write("Not eligible for further consideration.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
