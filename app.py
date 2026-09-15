import streamlit as st
import pandas as pd
import numpy as np
import pickle


@st.cache_resource
def load_artifacts():

    with open("xgboost_demand_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("labelencoder.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


model, label_encoders = load_artifacts()


st.title("Demand Forecusting App")

st.divider()


st.header('Input Feature')


price=st.number_input("Price", min_value=0.0,value=50.00)
discount=st.number_input("Discount",min_value=0.0,max_value=100.00,value=10.0)
inventory_level=st.number_input("Inventory Level",min_value=0.0,value=100.0)
promotion=st.selectbox("Promotion", [0,1])
competitor_price=st.number_input("Competitor Price",min_value=0.0,value=100.0)

category=st.selectbox(
    'Category',
    label_encoders['Category'].classes_.tolist()

)


input_data=pd.DataFrame({
    'Price' : [price],
    'Discount':[discount],
    'Inventory Level':[inventory_level],
    'Promotion':[promotion],
    'Competitor Pricing' :[competitor_price],
    'Category' :[category]
})

for col,encoder in label_encoders.items():
    if col in input_data.columns:
        input_data[col]=encoder.transform(input_data[col])

st.divider()
st.subheader("Input Data")
st.dataframe(input_data)

if st.button("Predict Demand"):
    try:

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Demand: {int(prediction)} Units")
    except Exception as e:

        st.error(f"Prediction Error: {e}")    

