import streamlit as st
import pickle
import os

# Get the absolute path to the pickled model file
model_path = os.path.join(os.getcwd(), 'D:/Deep_Project1/mpg_regression.sav')

st.title('Mpg ML Project')

displacement = st.number_input("displacement",value=307,placeholder='enter a value for displacement')
horsepower = st.number_input("Horsepower",value=130,placeholder='enter a value for horsepower')
weight = st.number_input("Weight",value=3507,placeholder='enter a value for weight')
accelaration = st.number_input("Accelaration",value=12,placeholder='enter a value for accelaration')

loaded_model=pickle.load(open('mpg_regression.sav','rb'))
prediction = loaded_model.predict([[displacement,horsepower,weight,accelaration]])
st.subheader(f'Predicted MPG value for above parameter is {prediction[0]}')
st.write(prediction)
