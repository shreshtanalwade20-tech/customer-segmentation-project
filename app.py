import streamlit as st
import joblib
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

st.title("Customer Segmentation & Purchase Prediction")

# Dataset
df = pd.read_csv("data/mall_customers.csv")

# Models
kmeans = joblib.load("models/kmeans_model.pkl")
svm = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.info("Model Used: K-Means Clustering + SVM Classification")

# Inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Annual Income",
    min_value=0,
    max_value=500,
    value=50
)

frequency = st.number_input(
    "Purchase Frequency",
    min_value=0,
    max_value=100,
    value=10
)

days = st.number_input(
    "Last Purchase Days",
    min_value=0,
    max_value=365,
    value=30
)

if income > 139:
    st.warning(
        "Income exceeds training data range (15-139). Prediction may be less accurate."
    )

if st.button("Analyze Customer"):

    customer = np.array([
        [age,
         income,
         frequency,
         days]
    ])

    customer_scaled = scaler.transform(customer)

    cluster = kmeans.predict(
        customer_scaled
    )[0]

    buyer = svm.predict(
        customer_scaled
    )[0]

    st.subheader("Prediction Results")

    st.success(
        f"Customer Segment: {cluster}"
    )

    if buyer == 1:
        st.success("Likely Buyer: YES")
    else:
        st.error("Likely Buyer: NO")

    # ---------- CLUSTERS ----------

    features = df[[
        'Age',
        'AnnualIncome',
        'PurchaseFrequency',
        'LastPurchaseDays'
    ]]

    scaled_features = scaler.transform(features)

    df['Cluster'] = kmeans.predict(
        scaled_features
    )

    # ---------- SCATTER GRAPH ----------

    st.subheader(
        "Income vs Purchase Frequency"
    )

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
        title="Customer Segments"
    )

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

    # ---------- SEGMENT BAR CHART ----------

    st.subheader(
        "Customer Segment Distribution"
    )

    segment_counts = (
        df['Cluster']
        .value_counts()
        .sort_index()
    )

    st.bar_chart(segment_counts)

    # ---------- BUYER PIE CHART ----------

    st.subheader(
        "Likely Buyers vs Non-Buyers"
    )

    buyer_counts = (
        df['LikelyBuyer']
        .value_counts()
    )

    pie_chart = go.Figure(
        data=[
            go.Pie(
                labels=[
                    "Not Buyer",
                    "Likely Buyer"
                ],
                values=buyer_counts.values
            )
        ]
    )

    st.plotly_chart(
        pie_chart,
        use_container_width=True
    )

    # ---------- PRODUCT CATEGORY ----------

    st.subheader(
        "Product Category Distribution"
    )

    category_counts = (
        df['ProductCategory']
        .value_counts()
    )

    st.bar_chart(
        category_counts
    )

    # ---------- ACCURACY ----------

    st.subheader(
        "Model Performance"
    )

    st.success(
        "SVM Accuracy: Update with your train.py result"
    )