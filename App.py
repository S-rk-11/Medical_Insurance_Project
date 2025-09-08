import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("medical_insurance_model.pkl", "rb") as file:
    model = pickle.load(file)

# App title & Description
st.set_page_config(page_title="Medical Insurance Predictor", layout="centered")

st.title("Medical Insurance Prediction")
st.markdown(
    """
    This app predicts **medical insurance costs** based on user details.
    Please fill in the form below and click **Predict".
    """
)

st.markdown("---")

# Sidebar for User inputs
st.sidebar.header("User Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

st.markdown("---")
            
# Data Preprocessing (encoding)
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_dict = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region = region_dict[region]

# Prepare features
features = np.array([[age, bmi, children, sex, smoker, region]])

# Predict button
if st.button("Predict Medical Cost"):
    prediction = model.predict(features)
    st.success(f"Predicted Medical Insurance Cost: ${prediction[0]:,.2f}")
