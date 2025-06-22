# ANN-CHURN
# 🧠 Customer Churn Prediction using Artificial Neural Networks (ANN)

This project is an interactive web application that predicts whether a customer is likely to churn based on key behavioral and financial features.

Built using:
- ✅ TensorFlow/Keras for model training and prediction  
- ✅ Scikit-learn for preprocessing (scaling, encoding)  
- ✅ Streamlit for creating a clean, interactive UI  
- ✅ Pickle for loading saved encoders and scalers  

---

## 🔍 Features

- Predicts churn likelihood using an ANN model trained on the **Churn_Modelling.csv** dataset.
- User-friendly UI to input customer attributes like:
  - Geography, Gender, Age, Credit Score
  - Balance, Salary, Tenure, Number of Products
  - Credit Card ownership, Active Membership
- Real-time prediction output with churn probability.
- Pre-trained model and preprocessing pipeline are included for instant use.

---

## 📂 Key Files

| File                      | Description                             |
|---------------------------|-----------------------------------------|
| `app.py`                  | Streamlit app code                      |
| `model.h5`                | Trained ANN model                       |
| `scaler.pkl`              | Scaler for input normalization          |
| `label_encode_gender.pkl` | LabelEncoder for gender                 |
| `onehot_geo.pkl`          | OneHotEncoder for geography             |
| `Churn_Modelling.csv`     | Original dataset                        |
| `requirements.txt`        | Python dependencies                     |
| `experiments.ipynb`       | Model training and experimentation      |
| `prediction.ipynb`        | Testing predictions with trained model  |

---

## 🚀 Getting Started

1. Clone the repo  
2. Install dependencies  
3. Run the Streamlit app using:

```bash
streamlit run app.py
