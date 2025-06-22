import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load model and preprocessing objects
model = tf.keras.models.load_model('model.h5')

with open('label_encode_gender.pkl', 'rb') as file:
    label_encode_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehot_geo.pkl', 'rb') as file:
    onehot_geo = pickle.load(file)

# Streamlit app title
st.title('💡 Customer Churn Prediction App')

st.markdown("Fill the details below to predict whether the customer is likely to churn.")

# User inputs
geography = st.selectbox('🌍 Geography', onehot_geo.categories_[0])
gender = st.selectbox('⚧ Gender', label_encode_gender.classes_)
age = st.slider('🎂 Age', 18, 92, 30)
credit_score = st.number_input('💳 Credit Score', value=650)
balance = st.number_input('💰 Balance', value=0.0)
estimated_salary = st.number_input('💼 Estimated Salary', value=50000.0)
tenure = st.slider('⌛ Tenure (Years)', 0, 10, 3)
num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])

# Process input
gender_encoded = label_encode_gender.transform([gender])[0]
geo_encoded = onehot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))

# Create input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Combine with one-hot geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Output
st.subheader("🧠 Prediction Result:")
if prediction_proba > 0.5:
    st.error(f"The customer is **likely to churn**. 🔻\n\n**Probability: {prediction_proba:.2f}**")
else:
    st.success(f"The customer is **not likely to churn**. ✅\n\n**Probability: {prediction_proba:.2f}**")
