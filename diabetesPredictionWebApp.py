#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:26:57 2024

@author: sumit
"""

import numpy as np
import pickle
import streamlit as st
# loading the saved model
loaded_model = pickle.load(open('/Users/sumit/Desktop/project/trained_model.sav', 'rb'))


def diabetesPrediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

def main():
    
    st.title('Diabetes Prediction Web App')
    
    #getting input data from user
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value')
    Age = st.text_input('enter you age')
    
    
    
    #prediction
    diagnosis = ''
    
    #creating button for predicition
    if st.button('Diabetes Test Result'):
        diagnosis = diabetesPrediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    