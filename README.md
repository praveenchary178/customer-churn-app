# 📊 Customer Churn Prediction App

A full-stack Machine Learning web application that predicts whether a customer is likely to churn or stay, built using **XGBoost** and deployed with **Streamlit**.

---

## 🚀 Live Demo
🔗https://your-app-link.onrender.com ](https://customer-churn-app-iu1h.onrender.com

---

## 🧠 Project Overview

Customer churn is a critical problem in business analytics. This project builds a predictive model to identify customers at risk of leaving and provides actionable insights using visualization and feature importance.

---

## ⚙️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Machine Learning:** XGBoost, Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Deployment:** Render  

---

## 📌 Features

- 🔮 Real-time churn prediction  
- 📊 Probability visualization (Churn vs Stay)  
- 📈 Feature importance analysis  
- ⚠️ Risk classification (Low / Medium / High)  
- 🎯 Clean and interactive UI  
- 🚀 Fully deployed web application  

---

## 🗂️ Project Structure
├── app.py # Streamlit application
├── model.pkl # Trained ML model
├── scaler.pkl # Data scaler
├── requirements.txt # Dependencies
└── README.md # Project documentation



---

## 📊 Model Details

- Algorithm: **XGBoost Classifier**
- Dataset: Kaggle Telco Customer Churn Dataset  
- Evaluation Metrics:
  - ROC-AUC Score: ~0.87  
  - Improved F1-score using SMOTE  
- Feature Engineering:
  - One-hot encoding  
  - Feature scaling  
  - Class imbalance handling  

---

## 🧪 How to Run Locally

### 1️⃣ Clone Repository
bash
git clone https://github.com/your-username/customer-churn-app.git
cd customer-churn-app
pip install -r requirements.txt
streamlit run app.py
