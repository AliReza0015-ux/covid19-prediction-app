# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from clustering import (
    load_data,
    train_kmeans,
    calculate_wcss,
    calculate_silhouette
)

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title(" Mall Customer Segmentation using KMeans Clustering")

# Upload or load default data
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom file uploaded.")
else:
    df = load_data("data/mall_customers.csv")
    st.info("Using default dataset from /data/mall_customers.csv")

st.subheader(" Data Preview")
st.dataframe(df.head())

# Feature selection
numerical_features = df.select_dtypes(include='number').columns.tolist()
features = st.multiselect("Select features for clustering", numerical_features,
                          default=['Annual_Income', 'Spending_Score'])

# k selection
k = st.slider("Choose number of clusters (k)", min_value=2, max_value=10, value=5)

# Train and predict
if len(features) >= 2:
    model = train_kmeans(df, features, k)
    df['Cluster'] = model.labels_

    st.subheader(" Cluster Visualization")
    if len(features) == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='Set2', ax=ax)
        plt.title("Customer Segments")
        st.pyplot(fig)
    else:
        st.info("Please select exactly 2 features to see cluster plot.")

    # Elbow Plot
    st.subheader("Elbow Method")
    k_range = range(2, 11)
    wcss = calculate_wcss(df, features, k_range)
    fig2, ax2 = plt.subplots()
    ax2.plot(k_range, wcss, marker='o')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("WCSS")
    ax2.set_title("Elbow Plot")
    st.pyplot(fig2)

    # Silhouette Plot
    st.subheader("Silhouette Score")
    sil_scores = calculate_silhouette(df, features, k_range)
    fig3, ax3 = plt.subplots()
    ax3.plot(k_range, sil_scores, marker='o', color='green')
    ax3.set_xlabel("Number of Clusters (k)")
    ax3.set_ylabel("Silhouette Score")
    ax3.set_title("Silhouette Plot")
    st.pyplot(fig3)
else:
    st.warning("Please select at least 2 numerical features.")
