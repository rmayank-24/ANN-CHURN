import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Churn Predictor", layout="wide")

# Load model and preprocessing objects
model = tf.keras.models.load_model('model.h5')

with open('label_encode_gender.pkl', 'rb') as file:
    label_encode_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehot_geo.pkl', 'rb') as file:
    onehot_geo = pickle.load(file)

# App title and intro
st.title('💡 Customer Churn Prediction App')
st.markdown("Predict whether a customer is likely to **churn** based on their banking profile.")

st.markdown("---")

# User inputs
st.header("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_geo.categories_[0])
    gender = st.selectbox('⚧ Gender', label_encode_gender.classes_)
    age = st.slider('🎂 Age', 18, 92, 30)
    credit_score = st.number_input('💳 Credit Score', value=650)
    balance = st.number_input('💰 Balance', value=0.0)
with col2:
    estimated_salary = st.number_input('💼 Estimated Salary', value=50000.0)
    tenure = st.slider('⌛ Tenure (Years)', 0, 10, 3)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
    is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])

# Threshold selection
threshold = st.slider("⚙️ Custom Decision Threshold", 0.0, 1.0, 0.5)

# Encode inputs
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
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Result
st.markdown("---")
st.subheader("🧠 Prediction Result")
st.progress(prediction_proba)

if prediction_proba > threshold:
    st.error(f"🔻 The customer is **likely to churn**.\n\n**Probability: {prediction_proba:.2f}**")
else:
    st.success(f"✅ The customer is **not likely to churn**.\n\n**Probability: {prediction_proba:.2f}**")

# Input summary sidebar
with st.sidebar:
    st.header("📝 Input Summary")
    st.write(f"**Geography**: {geography}")
    st.write(f"**Gender**: {gender}")
    st.write(f"**Age**: {age}")
    st.write(f"**Credit Score**: {credit_score}")
    st.write(f"**Balance**: {balance}")
    st.write(f"**Estimated Salary**: {estimated_salary}")
    st.write(f"**Tenure**: {tenure}")
    st.write(f"**Number of Products**: {num_of_products}")
    st.write(f"**Has Credit Card**: {'Yes' if has_cr_card else 'No'}")
    st.write(f"**Active Member**: {'Yes' if is_active_member else 'No'}")

# SHAP interpretability (optional)
if st.checkbox("🔍 Show SHAP Explanation"):
    with st.spinner("Generating SHAP explanation..."):
        try:
            explainer = shap.KernelExplainer(model.predict, input_scaled)
            shap_values = explainer.shap_values(input_scaled)
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value[0],
                data=input_scaled[0],
                feature_names=input_data.columns
            ), show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning("SHAP explanation could not be generated. Try running locally or in notebook.")
            st.text(f"Error: {e}")

# About
st.markdown("---")
st.markdown("""
📌 **About this App**  
Built using a trained Artificial Neural Network (ANN) model on a banking dataset.  
It predicts customer churn using key features like credit score, tenure, and activity.

👨‍💻 Developed by [Your Name](mailto:your@email.com)  
🔗 Source Code: [GitHub Repo](https://github.com/your-username/ANN-CHURN)  
🌐 Live Demo: [Streamlit App](https://ann-churn-rmayank.streamlit.app/)
""")
