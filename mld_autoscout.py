# Streamlit Documentation: https://docs.streamlit.io/


import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image  # to deal with images (PIL: Python imaging library)

# Title/Text
st.set_page_config(
    page_title="ML Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.markdown('<p class="description">Based on Autoscout Model</p>', unsafe_allow_html=True)




# Add select box
occupation=st.selectbox("Your Occupation", ["Programmer", "DataScientist", "Doctor"])
st.write("Your Occupation is ", occupation)

# Multi_select
multi_select = st.multiselect("Horsepower kW",[1,2,3,4,5])
st.write(f"You selected {len(multi_select)} number(s)")
st.write("Your selection is/are", multi_select)
for i in range(len(multi_select)):
    st.write(f"Your {i+1}. selection is {multi_select[i]}")

# Slider
option1 = st.slider("Horsepower kW", min_value=300, max_value=5000, value=300, step=5)
option2 = st.number_input("Gears", min_value=5, max_value=7)
option3 = st.number_input("Age", min_value=5, max_value=7)


# result=option1*option2
# st.write("multiplication of two options is:",result)

# Text_input
name = st.text_input("Enter your name", placeholder="Your name here")
if st.button("Submit"):
    st.write("Hello {}".format(name.title()))
    




# Dataframe
df=pd.read_csv("mld-autoscout-model.csv")

# To display dataframe there are 3 methods

# Method 1
st.table(df.head())
# Method 2
st.write(df.head())  # dynamic, you can sort
st.write(df.isnull().sum())
# Method 3
st.dataframe(df.describe().T)  # dynamic, you can sort

# To load machine learning model
import pickle
filename = "mld-autoscout-model.pkl"
model=pickle.load(open(filename, "rb"))

# To take feature inputs
hp_Kw = st.sidebar.number_input("Horsepower kW:",min_value=300, max_value=5000)
Gears = st.sidebar.number_input("Gears:",min_value=4, max_value=7)
age = st.sidebar.number_input("age:",min_value=0, max_value=50)
Weight_kg = st.sidebar.number_input("Weight in Kg:",min_value=300, max_value=5000)
km = st.sidebar.number_input("Km:",min_value=4, max_value=7)
Displacement_cc = st.sidebar.number_input("Displacement in CC:",min_value=0, max_value=50)
body_type = st.sidebar.number_input("Body Type:",min_value=0, max_value=50)
Fuel= st.sidebar.number_input("Fuel:",min_value=0, max_value=50)
Gearing_Type = st.sidebar.number_input("Gearing Type:",min_value=0, max_value=50)
Drive_chain = st.sidebar.number_input("Drive chain:",min_value=0, max_value=50)

# Create a dataframe using feature inputs
my_dict = {
            "hp_kW":hp_Kw,
            "Gears":Gears,
            "age":age,
            "Weight_kg":Weight_kg,
            "km":km,
            "Displacement_cc":Displacement_cc,
            "body_type":body_type,
            "Fuel":Fuel,
            "Gearing_Type":Gearing_Type,
            "Drive_chain":Drive_chain
           }

df = pd.DataFrame.from_dict([my_dict])
st.table(df)

# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(df)
if predict :
    st.success(result[0])