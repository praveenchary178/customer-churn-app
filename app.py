import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    text-align: center;
}
.stButton>button {
    width: 100%;
    background-color: #00c6ff;
    color: black;
    font-size: 18px;
    border-radius: 10px;
}
.card {
    background-color: #1e2a38;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

# ---------------- HEADER ----------------
st.markdown("""
<h1>📊 Customer Churn Prediction</h1>
<p style='text-align:center;'>Predict whether a customer will leave or stay</p>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1,1])

# ---------------- INPUT FORM ----------------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Enter Customer Details")

    with st.form("input_form"):
        credit_score = st.number_input("Credit Score", 300, 900)
        age = st.number_input("Age", 18, 100)
        tenure = st.number_input("Tenure", 0, 10)
        balance = st.number_input("Balance", 0.0)
        num_products = st.number_input("Number of Products", 1, 4)
        has_card = st.selectbox("Has Credit Card", [0,1])
        is_active = st.selectbox("Is Active Member", [0,1])
        salary = st.number_input("Estimated Salary", 0.0)

        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        submit = st.form_submit_button("Predict 🚀")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if submit:

        # Encoding
        geo_germany = 1 if geography == "Germany" else 0
        geo_spain = 1 if geography == "Spain" else 0
        gender_male = 1 if gender == "Male" else 0

        # Input
        input_data = np.array([[credit_score, age, tenure, balance,
                                num_products, has_card, is_active, salary,
                                geo_germany, geo_spain, gender_male]])

        input_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # ---------------- RESULT ----------------
        if prediction == 1:
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Customer Likely to Stay")

        st.markdown(f"### 📊 Probability of Churn: {probability*100:.2f}%")
        st.progress(int(probability * 100))

        # ---------------- RISK LEVEL ----------------
        if probability > 0.7:
            st.warning("🚨 Very High Risk Customer")
        elif probability > 0.4:
            st.info("⚠️ Medium Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

        # ---------------- PROBABILITY GRAPH ----------------
        st.subheader("📊 Prediction Probability")

        prob_df = pd.DataFrame({
            "Class": ["Stay", "Churn"],
            "Probability": [1 - probability, probability]
        })

        fig, ax = plt.subplots()
        ax.bar(prob_df["Class"], prob_df["Probability"])
        ax.set_ylabel("Probability")
        ax.set_title("Churn vs Stay")

        st.pyplot(fig)

        # ---------------- FEATURE IMPORTANCE ----------------
        st.subheader("📈 Feature Importance")

        features = ["CreditScore", "Age", "Tenure", "Balance",
                    "NumProducts", "HasCard", "IsActive",
                    "Salary", "Geo_Germany", "Geo_Spain", "Gender_Male"]

        importance = model.feature_importances_

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig2, ax2 = plt.subplots()
        ax2.barh(imp_df["Feature"], imp_df["Importance"])
        ax2.set_title("Feature Importance")

        st.pyplot(fig2)

        # ---------------- MODEL INSIGHT ----------------
        st.subheader("🧠 Model Insight")

        top_feature = imp_df.iloc[0]["Feature"]
        st.write(f"🔍 Most important factor influencing churn: **{top_feature}**")

    else:
        st.info("Fill the details and click Predict")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;'>Built with ❤️ using Streamlit | XGBoost Model</p>
""", unsafe_allow_html=True)