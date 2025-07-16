import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load model and preprocessing objects
model = tf.keras.models.load_model('model.h5')
with open('label_encode_gender.pkl', 'rb') as file:
    label_encode_gender = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('onehot_geo.pkl', 'rb') as file:
    onehot_geo = pickle.load(file)

# Load dataset for dashboard
df = pd.read_csv("Churn_Modelling.csv")
df['Gender'] = label_encode_gender.transform(df['Gender'])
geo_encoded = onehot_geo.transform(df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))
df = pd.concat([df.drop(['Geography', 'RowNumber', 'CustomerId', 'Surname'], axis=1), geo_encoded_df], axis=1)

# Create Tabs
tab1, tab2 = st.tabs(["📊 Dashboard", "🤖 Prediction"])

with tab1:
    st.title("📊 Data Dashboard")
    st.markdown("Exploratory Data Insights from the Churn Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        churn_rate = df['Exited'].mean()
        st.metric("Churn Rate", f"{churn_rate:.2%}")
    with col2:
        avg_salary = df['EstimatedSalary'].mean()
        st.metric("Avg Salary", f"${avg_salary:,.2f}")
    with col3:
        avg_age = df['Age'].mean()
        st.metric("Avg Age", f"{avg_age:.1f} years")

    st.subheader("🎯 Target Variable Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Exited', data=df, palette='Set2', ax=ax1)
    ax1.set_xticklabels(['Not Churned', 'Churned'])
    st.pyplot(fig1)

    st.subheader("📌 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📊 Feature Distributions")
    for col in ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, bins=30, color='skyblue')
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

with tab2:
    st.title("🤖 Customer Churn Prediction App")
    st.markdown("Fill the details below to predict whether the customer is likely to churn.")

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
    geo_encoded_input = onehot_geo.transform([[geography]]).toarray()
    geo_encoded_input_df = pd.DataFrame(geo_encoded_input, columns=onehot_geo.get_feature_names_out(['Geography']))

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

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_input_df], axis=1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_proba = float(prediction[0][0])  # ✅ Convert float32 to Python float

    st.markdown("---")
    st.subheader("🧠 Prediction Result")
    st.progress(prediction_proba)
    st.caption(f"🔍 Probability of churn: {prediction_proba:.2%}")

    if prediction_proba > 0.5:
        st.error(f"🔻 The customer is **likely to churn**.\n\n**Probability: {prediction_proba:.2f}**")
    else:
        st.success(f"✅ The customer is **not likely to churn**.\n\n**Probability: {prediction_proba:.2f}**")
