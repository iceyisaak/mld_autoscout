# Streamlit Documentation: https://docs.streamlit.io/


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image  # to deal with images (PIL: Python imaging library)
import pickle
import os 

# Title/Text
st.set_page_config(
    page_title="ML Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("mld-autoscout-model.csv")
        return df
    except:
        # Generate sample data if file doesn't exist
        data = {
            'TV': np.random.uniform(10, 300, 200),
            'radio': np.random.uniform(1, 50, 200),
            'newspaper': np.random.uniform(0, 120, 200),
            'sales': np.random.uniform(1, 30, 200)
        }
        df = pd.DataFrame(data)
        return df


# Load model function with proper caching and error handling
@st.cache_resource
def load_model():
    model_path = "mld-autoscout-model.pkl"
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


df = load_data()
model = load_model()


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4267B2;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
    }
    .description {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🚗 ML Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Based on Autoscout Dataset</p>', unsafe_allow_html=True)

st.subheader("Enter the Input for Price Prediction")

col1, col2 = st.columns(2)

with col1: 
    hp_kW = st.slider("Horsepower kW", min_value=int(df['hp_kW'].min()), max_value=int(df['hp_kW'].max()), value=100, step=5)
    Gears = st.slider(
            "Gears", 
            min_value=int(df['Gears'].min()), 
            max_value=int(df['Gears'].max()), 
            value=int(df['Gears'].min()), 
            step=1
        )
    age = st.slider("Age", min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=1, step=1)
    body_type = st.selectbox("body_type", df.body_type.unique().tolist())
    Gearing_Type = st.selectbox("Transmission", df.Gearing_Type.unique().tolist())

with col2:
    km = st.slider("Km", min_value=int(df['km'].min()), max_value=int(df['km'].max()), value=300, step=10)
    Displacement_cc = st.slider(
            "Displacement CC", 
            min_value=int(df['Displacement_cc'].min()), 
            max_value=int(df['Displacement_cc'].max()), 
            value=int(df['Displacement_cc'].min()), 
            step=10
        )
    Weight_kg = st.slider(
            "Weight_kg", 
            min_value=int(df['Weight_kg'].min()), 
            max_value=int(df['Weight_kg'].max()), 
            value=int(df['Weight_kg'].min()), 
            step=1
        )
    Drive_chain = st.selectbox("Drive_chain", df.Drive_chain.unique().tolist())
    Fuel = st.selectbox("Fuel", df.Fuel.unique().tolist())



st.subheader("Overview of Sample Data")
# Data Overview
st.table(df.head())


# Stat Summary
st.subheader("Statistical Summary")
st.dataframe(df.describe().T)  # dynamic, you can sort


input_data = pd.DataFrame({
            'hp_kW': [hp_kW],
            'Gears': [Gears],
            'Gearing_Type': [Gearing_Type],
            'age':[age],
            'km':[km],
            'body_type':[body_type],
            'Fuel':[Fuel],
            'Drive_chain':[Drive_chain],
            'Weight_kg':[Weight_kg],
            'Displacement_cc':[Displacement_cc]
        })



# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(input_data)
if predict :
    # st.success(result[0])
    st.success(f"Predicted Price: ${result[0]:.2f}")
    