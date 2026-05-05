import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("📊 Customer Segmentation using RFM & K-Means")

# Upload file
file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if file is not None:

    # Read file
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='ISO-8859-1')
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.subheader("🔹 Raw Data Preview")
    st.dataframe(df.head())

    # =========================
    # PREPROCESSING
    # =========================
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # =========================
    # RFM CALCULATION
    # =========================
    snapshot = df['InvoiceDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # Remove outliers
    rfm = rfm[
        (rfm['Monetary'] < rfm['Monetary'].quantile(0.99)) &
        (rfm['Frequency'] < rfm['Frequency'].quantile(0.99))
    ]

    st.subheader("🔹 RFM Table")
    st.dataframe(rfm.head())

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # =========================
    # K-MEANS
    # =========================
    st.subheader("🔹 Silhouette Scores")

    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(rfm_scaled)
        from sklearn.metrics import silhouette_score
        score = silhouette_score(rfm_scaled, labels)
        scores[k] = score

    st.write(scores)

    # Final model
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # =========================
    # SEGMENT NAMING
    # =========================
    def assign_segment(row):
        r, f = row['Recency'], row['Frequency']

        if r < 30 and f > 10:
            return "Loyal High-Spenders"
        elif r > 70 and f < 5:
            return "At-Risk Infrequents"
        elif 5 <= f <= 10:
            return "Regular Customers"
        else:
            return "New/Occasional Customers"

    cluster_map = rfm.groupby('Cluster').mean().apply(assign_segment, axis=1)
    rfm['Segment'] = rfm['Cluster'].map(cluster_map)

    # =========================
    # OUTPUT
    # =========================
    st.subheader("🔹 Final Segmented Data")
    st.dataframe(rfm.head())

    # Download
    csv = rfm.to_csv().encode('utf-8')
    st.download_button("📥 Download Segmented Data", csv, "customer_segments.csv")

else:
    st.info("Please upload a dataset to begin.")