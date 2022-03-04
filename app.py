import streamlit as st
import pandas as pd
import joblib

string = "Machine Learning"
st.set_page_config(page_title=string, page_icon="😆")

st.title("Machine Learning")
st.write("""
# Salary Prediction Model
Salary vs. *Experience*
""")

exp = st.sidebar.slider('Experience',1.0,50.0,2.0)
reg = joblib.load("salary.pkl")
y_pred = reg.predict([[exp]])

st.write(f"Experience: ", exp)
st.write(f"Salary: ", float(y_pred))