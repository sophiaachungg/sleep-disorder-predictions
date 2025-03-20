import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load the model
model = joblib.load('sleep_disorder_model.joblib')

# BMI calculation function
def calculate_bmi(weight, height, unit):
    if unit == 'metric':
        bmi = weight / ((height / 100) ** 2)
    else:
        bmi = (weight / (height ** 2)) * 703
    return round(bmi, 2)

def calculate_map(systolic_bp, diastolic_bp):
    map = 1/3 * systolic_bp + 2/3 * diastolic_bp
    return map

# create Streamlit interface
st.title('Sleep Disorder Prediction')

# add input fields for user data
# BMI-specific code
unit = st.radio('Select unit system:', ('Metric', 'Imperial'))
if unit == 'Metric':
    weight = st.number_input('Enter weight (kg):', min_value=30, max_value=300, value=70)
    height = st.number_input('Enter height (cm):', min_value=100, max_value=250, value=170)
    unit_system = 'metric'
else:
    weight = st.number_input('Enter weight (lb):', min_value=65, max_value=500, value=150)
    height = st.number_input('Enter height (in):', min_value=40, max_value=110, value=65)
    unit_system = 'imperial'
bmi = calculate_bmi(weight, height, unit)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
sleep_duration = st.number_input('Average Sleep Duration', min_value=1, max_value=14, value=7)
physical_activity = st.slider('How would you rate your physical activity level?', 1, 5, 3)
# use formula to compute physical_activity: 30 + (old_value - 1) * (90 - 30) / (5 - 1)
physical_activity = 30 + (physical_activity-1) * 60 / 4
systolic_bp = st.number_input('Systolic Blood Pressure', min_value=90, max_value=200, value=120)
diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=60, max_value=120, value=80)
map = calculate_map(systolic_bp, diastolic_bp)

# assign BMI category
if bmi < 25:
    bmi_category = 'Normal'
    bmi_norm = 1
    bmi_over = 0
else:
    bmi_category = 'Overweight'
    bmi_norm = 0
    bmi_over = 1

# create a dataframe from user input
user_data = pd.DataFrame({
    'BMI Category_Normal': [bmi_norm],
    'BMI Category_Overweight': [bmi_over],
    'Occupation_Nurse': 0,
    'Mean Arterial Pressure': [map],
    'Sleep Duration': [sleep_duration],
    'Age': [age],
    'Physical Activity Level': [physical_activity],
})

# load scaler
scaler = joblib.load('scaler.joblib')

# standardize user_data
continuous_col = ['Mean Arterial Pressure', 'Sleep Duration', 'Age', 'Physical Activity Level']
binary_col = ['BMI Category_Normal', 'BMI Category_Overweight', 'Occupation_Nurse']
user_data_std = scaler.transform(user_data[continuous_col])
df = pd.DataFrame(user_data_std, columns=continuous_col)
df = pd.concat([user_data[binary_col], df], axis=1)

# make predictions
if st.button('Predict'):
    prediction = model.predict(df)
    messages = ['You likely have no sleep disorders. Keep up the healthy habits.',
                'Based on your health and lifestyle, you likely have sleep apnea. See a doctor for a proper diagnosis.', 
                'Based on your health and lifestyle, you likely have insomnia. See a doctor for a proper diagnosis.']
    output = messages[prediction[0]]
    st.write(f'{output}')
    
    # display top 2 features that are contributing to potential sleep disorder
    if prediction[0] != 0:
        # Get feature importances
        feature_importances = model.feature_importances_
        feature_names = user_data.columns
        
        # Create a dataframe of feature importances
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Filter out 'BMI Category_Normal' and 'Occupation_Nurse'
        filtered_importance = importance_df[~importance_df['feature'].isin(['BMI Category_Normal', 'Occupation_Nurse'])]
        
        # Get top 2 features
        top_features = filtered_importance.head(2)
        
        st.subheader("Top 2 Contributing Factors:")
        for index, row in top_features.iterrows():
            st.write(f"{row['feature']}")
        
        st.write("Consider addressing these factors to improve your sleep health.")
    
        # Add interpretations for each feature
        feature_interpretations = {
            'Age': "Age can impact sleep patterns and disorder risk.",
            'Sleep Duration': "Both too little and too much sleep can be indicators of sleep issues.",
            'Physical Activity Level': "Regular physical activity can improve sleep quality.",
            'Mean Arterial Pressure': "Blood pressure can be both a cause and effect of sleep disorders.",
            'BMI Category_Overweight': "Being overweight can increase the risk of certain sleep disorders."
        }
    
        for feature in top_features['feature']:
            if feature in feature_interpretations:
                st.write(feature_interpretations[feature])
        st.write('DISCLAIMER: This is not an official medical diagnosis. Please see your doctor if you have any concerns.')
