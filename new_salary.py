import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
import numpy as np
import pickle
import streamlit as st

# loading the saved model
#loaded_model_diabetes = pickle.load(open('C:/Users/sujee/Downloads/mineproject/trained_model.sav', 'rb'))
#loaded_model_salary = pickle.load(open('C:/Users/sujee/Downloads/mineproject/linear_regression.pkl', 'rb'))
loaded_model_diabetes = pickle.load(open('trained_model.sav', 'rb'))
loaded_model_salary = pickle.load(open('linear_regression.pkl', 'rb'))

def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model_diabetes.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def predict_salary(loaded_model, input_features):
    try:
        # Convert input features to numeric values
        input_features = [float(feature) for feature in input_features]
    except ValueError:
        return "Please enter valid numeric values for all features."

    input_features = np.array(input_features).reshape(1, -1)

    # Predict the salary using the loaded model
    predicted_salary = loaded_model.predict(input_features)[0]

    return predicted_salary

def main():
    with st.sidebar:
        selected = option_menu('DIABETES and SALARYCTC PREDICTION',
                               ['DIABETES PREDICTION', 'SALARY PREDICTION', 'ITS_SU_RJ'],
                               icons=['activity', 'currency-rupee', 'emoji-heart-eyes'],
                               default_index=0)

    # diabetes prediction page
    if selected == 'DIABETES PREDICTION':
        st.title('Diabetes prediction using SVM')

        # getting the input data from user
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose level')
        BloodPressure = st.text_input('Blood pressure value')
        SkinThickness = st.text_input('Skin thickness value')
        Insulin = st.text_input('Insulin level')
        BMI = st.text_input('BMI value')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age value')

        # code for prediction
        diagnosis = ''

        # creating a button for prediction
        if st.button('Diabetes test result'):
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                              DiabetesPedigreeFunction, Age])

        st.success(diagnosis)

    # salary prediction page
    elif selected == 'SALARY PREDICTION':
        st.title('Salary CTC prediction using multiple linear regression')

        # getting the input data from the user
        grade = st.text_input('Grade (1-10)')
        softskill = st.text_input('Soft skill (1-10)')
        problemsolvingskill = st.text_input('Problem-solving skill (1-10)')
        meditationandyoga = st.text_input('Meditation and Yoga (1-10)')
        discipline = st.text_input('Discipline level (1-10)')
        strongcommandinoneskill = st.text_input('Strong command in one skill (1-10)')

        # code for prediction
        diagnosis = ''

        # creating a button for prediction
        if st.button('Predict Salary CTC in Lacs'):
            new_features = [grade, softskill, problemsolvingskill, meditationandyoga, discipline, strongcommandinoneskill]
            diagnosis = predict_salary(loaded_model_salary, new_features)

        # st.success(f'Predicted Salary CTC: {diagnosis:.2f} Lacs')
        st.success(f'Predicted Salary CTC: {float(diagnosis):.2f} Lacs' if isinstance(diagnosis, (int, float)) else diagnosis)

    # other page
    elif selected == 'ITS_SU_RJ':
        st.title('THANK YOU')

if __name__ == '__main__':
    main()
