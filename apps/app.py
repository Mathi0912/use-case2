import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
st.set_page_config(page_title="Customer Lifetime Value", layout="centered")

st.title("💰 Customer Lifetime Value Prediction")
st.write("Predict future customer value")
np.random.seed(42)

train_df = pd.DataFrame({
    "Membership_Years": np.random.randint(0, 10, 400),
    "Total_Purchases": np.random.randint(0, 200, 400),
    "Average_Order_Value": np.random.uniform(50, 500, 400),
    "Discount_Usage_Rate": np.random.uniform(0, 100, 400),
    "Returns_Rate": np.random.uniform(0, 50, 400)
})

train_df["Lifetime_Value"] = (
    train_df["Total_Purchases"] * train_df["Average_Order_Value"]
    - 40 * train_df["Returns_Rate"]
    - 20 * train_df["Discount_Usage_Rate"]
    + 150 * train_df["Membership_Years"]
)
X = train_df.drop("Lifetime_Value", axis=1)
y = train_df["Lifetime_Value"]
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
model.fit(X, y)
st.sidebar.header("Customer Inputs")

membership = st.sidebar.number_input("Membership Years", 0, 20, 2)
purchases = st.sidebar.number_input("Total Purchases", 0, 500, 50)
aov = st.sidebar.number_input("Average Order Value", 0.0, 1000.0, 150.0)
discount = st.sidebar.slider("Discount Usage Rate (%)", 0.0, 100.0, 20.0)
returns = st.sidebar.slider("Returns Rate (%)", 0.0, 100.0, 5.0)

input_df = pd.DataFrame({
    "Membership_Years": [membership],
    "Total_Purchases": [purchases],
    "Average_Order_Value": [aov],
    "Discount_Usage_Rate": [discount],
    "Returns_Rate": [returns]
})
if st.button("Predict Lifetime Value"):
    ltv = model.predict(input_df)[0]
    st.success(f"💵 Predicted Lifetime Value: {ltv:.2f}")
