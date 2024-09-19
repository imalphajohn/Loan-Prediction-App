import streamlit as st
import pickle5
import numpy as np
import sklearn

# Load the saved model
with open('loan_approval_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Loan Approval Prediction')

st.write('Please enter the following information:')

gender = st.radio('Gender', ['Male', 'Female'])
married = st.radio('Married', ['Yes', 'No'])
education = st.radio('Education', ['Graduate', 'Not Graduate'])
self_employed = st.radio('Self Employed', ['Yes', 'No'])
property_area = st.selectbox('Property Area', ['Urban', 'Semiurban', 'Rural'])

if st.button('Predict Loan Approval'):
    # Prepare the input data
    input_data = [
        1 if gender == 'Male' else 0,
        1 if married == 'Yes' else 0,
        1 if education == 'Not Graduate' else 0,
        1 if self_employed == 'Yes' else 0,
        1 if property_area == 'Semiurban' else 0,
        1 if property_area == 'Urban' else 0
    ]
    # Make prediction
    prediction = model.predict([input_data])
    
    # Display result
    if prediction[0] == 1:
        st.success('Loan is likely to be approved!')
    else:
        st.error('Loan is likely to be rejected.')# Display prediction probability
    probability = model.predict_proba([input_data])
    st.write(f'Probability of approval: {100*probability[0][1]:.2f}%')

st.write('Note: This is a simple model and should not be used for actual loan decisions.')
