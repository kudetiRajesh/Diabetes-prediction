#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# In[2]:


import streamlit as st
from joblib import load
import pandas as pd

# Load the trained model
from joblib import load

# Replace '/path/to/directory/model.pkl' with the full path to your saved model file
model = load('model_joblib.pkl')


# Define the function to predict diabetes
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level):
    # Convert gender to binary (e.g., Female=1, Male=0)
    gender = 1 if gender == 'Female' else 0
    
    smoking_history_map = {'never': 0, 'formerly': 1, 'currently': 2}
    smoking_history = smoking_history_map.get(smoking_history)
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({'gender': [gender],
                               'age': [age],
                               'hypertension': [hypertension],
                               'heart_disease': [heart_disease],
                               'smoking_history': [smoking_history],
                               'bmi': [bmi],
                               'HbA1c_level': [HbA1c_level],
                               'blood_glucose_level': [blood_glucose_level]})
    # Make predictions
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit UI
st.title('Diabetes Prediction App')

# Input fields
gender = st.selectbox('Gender', ['Female', 'Male'])
age = st.number_input('Age', min_value=0, max_value=150, value=30)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
smoking_history = st.selectbox('Smoking History', ['never', 'formerly', 'currently'])
bmi = st.slider('BMI', min_value=0.0, max_value=100.0, value=20.0)
HbA1c_level = st.slider('HbA1c Level', min_value=0.0, max_value=20.0, value=5.0)
blood_glucose_level = st.slider('Blood Glucose Level', min_value=0, max_value=500, value=100)

# Prediction button
if st.button('Predict'):
    # Call the prediction function
    prediction = predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level)
    # Display the prediction
    if prediction == 1:
        st.write('The patient is  have diabetes.')
    else:
        st.write('The patient is not have diabetes.')


# In[20]:




# In[25]:





# In[26]:




# In[33]:

from joblib import load





# In[ ]:





# In[ ]:





# In[53]:



# In[ ]:




