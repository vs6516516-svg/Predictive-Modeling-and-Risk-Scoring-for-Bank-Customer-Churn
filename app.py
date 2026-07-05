import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦")

st.title("🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn")
st.write("Predict whether a customer is likely to churn.")

# Load Dataset
df = pd.read_csv("European_Bank.csv")

# Drop unnecessary columns (only if present)
for col in ["RowNumber", "CustomerId", "Surname"]:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

# Features & Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

st.header("Enter Customer Details")

credit = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products = st.number_input("Number of Products", 1, 4, 1)
card = st.selectbox("Has Credit Card", [0, 1])
active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 300000.0, 100000.0)

geo = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

input_df = pd.DataFrame({
    "CreditScore":[credit],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[products],
    "HasCrCard":[card],
    "IsActiveMember":[active],
    "EstimatedSalary":[salary],
    "Geography_Germany":[1 if geo=="Germany" else 0],
    "Geography_Spain":[1 if geo=="Spain" else 0],
    "Gender_Male":[1 if gender=="Male" else 0]
})

input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = rf_model.predict(input_scaled)[0]
    probability = rf_model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"Customer is likely to CHURN\n\nRisk Score: {probability:.2%}")
    else:
        st.success(f"Customer is likely to STAY\n\nRisk Score: {probability:.2%}")
