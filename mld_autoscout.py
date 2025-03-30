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

# Load or create model function
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("mld-autoscout-model.pkl", "rb"))
        return model
    except:
        # Create dummy model if file doesn't exist
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        X = df[['TV', 'radio', 'newspaper']]
        y = df['sales']
        model.fit(X, y)
        return model

df = load_data()



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



# Slider
option1 = st.slider("Horsepower kW", min_value=int(df['hp_kW'].min()), max_value=int(df['hp_kW'].max()), value=100, step=5)
option2 = st.slider(
            "Gear", 
            min_value=int(df['Gears'].min()), 
            max_value=int(df['Gears'].max()), 
            value=int(df['Gears'].min()), 
            step=1
        )
option3 = st.slider("Age", min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=1, step=1)
option4 = st.slider("Km", min_value=int(df['km'].min()), max_value=int(df['km'].max()), value=300, step=10)
option5 = st.slider(
            "Displacement CC", 
            min_value=int(df['Displacement_cc'].min()), 
            max_value=int(df['Displacement_cc'].max()), 
            value=int(df['Displacement_cc'].min()), 
            step=10
        )
option6 = st.slider(
            "Weight_kg", 
            min_value=int(df['Weight_kg'].min()), 
            max_value=int(df['Weight_kg'].max()), 
            value=int(df['Weight_kg'].min()), 
            step=1
        )
option7 = st.selectbox("body_type", df.body_type.unique().tolist())
option8 = st.selectbox("Transmission", df.Gearing_Type.unique().tolist())
option9 = st.selectbox("Drive_chain", df.Drive_chain.unique().tolist())
option10 = st.selectbox("Fuel", df.Fuel.unique().tolist())

    





st.subheader("Overview of Sample Data")
# Data Overview
st.table(df.head())


# Stat Summary
st.subheader("Statistical Summary")
st.dataframe(df.describe().T)  # dynamic, you can sort



# # To take feature inputs
# hp_Kw = st.sidebar.number_input("Horsepower kW:",min_value=300, max_value=5000)
# Gears = st.sidebar.number_input("Gears:",min_value=4, max_value=7)
# age = st.sidebar.number_input("age:",min_value=0, max_value=50)
# Weight_kg = st.sidebar.number_input("Weight in Kg:",min_value=300, max_value=5000)
# km = st.sidebar.number_input("Km:",min_value=4, max_value=7)
# Displacement_cc = st.sidebar.number_input("Displacement in CC:",min_value=0, max_value=50)
# body_type = st.sidebar.number_input("Body Type:",min_value=0, max_value=50)
# Fuel= st.sidebar.number_input("Fuel:",min_value=0, max_value=50)
# Gearing_Type = st.sidebar.number_input("Gearing Type:",min_value=0, max_value=50)
# Drive_chain = st.sidebar.number_input("Drive chain:",min_value=0, max_value=50)

# Create a dataframe using feature inputs
my_dict = {
            # "hp_kW":hp_Kw,
            # "Gears":Gears,
            # "age":age,
            # "Weight_kg":Weight_kg,
            # "km":km,
            # "Displacement_cc":Displacement_cc,
            # "body_type":body_type,
            # "Fuel":Fuel,
            # "Gearing_Type":Gearing_Type,
            # "Drive_chain":Drive_chain
           }

df = pd.DataFrame.from_dict([my_dict])
st.table(df)

# Prediction with user inputs
# predict = st.button("Predict")
# result = model.predict(df)
# if predict :
#     st.success(result[0])