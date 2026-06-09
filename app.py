import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

st.title("Customer Segmentation & Purchase Prediction")

# Load dataset
df = pd.read_csv("data/mall_customers.csv")

# Dynamic limits from dataset
max_age = int(df["Age"].max())
max_income = int(df["AnnualIncome"].max())
max_frequency = int(df["PurchaseFrequency"].max())
max_days = int(df["LastPurchaseDays"].max())

# Load models
kmeans = joblib.load("models/kmeans_model.pkl")
svm = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# User Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=max_age,
    value=30
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    max_value=max_income,
    value=50
)

frequency = st.number_input(
    "Purchase Frequency",
    min_value=0,
    max_value=max_frequency,
    value=10
)

days = st.number_input(
    "Last Purchase Days",
    min_value=0,
    max_value=max_days,
    value=30
)

if st.button("Analyze Customer"):

    # Customer input
    customer = np.array([
        [age, income, frequency, days]
    ])

    # Scale customer data
    customer_scaled = scaler.transform(customer)

    # Predictions
    cluster = kmeans.predict(customer_scaled)[0]
    buyer = svm.predict(customer_scaled)[0]

    st.subheader("Prediction Results")

    st.success(f"Customer Segment: {cluster}")

    if buyer == 1:
        st.success("Likely Buyer: YES")
    else:
        st.error("Likely Buyer: NO")

    # Create clusters for all customers
    features = df[['Age',
                   'AnnualIncome',
                   'PurchaseFrequency',
                   'LastPurchaseDays']]

    scaled_features = scaler.transform(features)

    df['Cluster'] = kmeans.predict(scaled_features)

    st.subheader("Customer Segmentation Graph")

    # Scatter Plot
    fig = px.scatter(
        df,
        x="AnnualIncome",
        y="PurchaseFrequency",
        color=df["Cluster"].astype(str),
        hover_data=[
            "Age",
            "LastPurchaseDays",
            "ProductCategory",
            "LikelyBuyer"
        ],
        title="Customer Segments Based on Income and Purchase Frequency"
    )

    # Highlight current customer
    fig.add_scatter(
        x=[income],
        y=[frequency],
        mode="markers",
        marker=dict(
            size=18,
            symbol="star"
        ),
        name="Current Customer"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )