import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation using RFM & K-Means")

# =========================
# CACHE (Performance)
# =========================
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding='ISO-8859-1')
    else:
        return pd.read_excel(file)

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if file is not None:

    try:
        df = load_data(file)
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
    # RFM SCORING (1–5)
    # =========================
    rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_score'] = pd.qcut(rfm['Frequency'], 5, labels=[1,2,3,4,5])
    rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

    rfm['RFM_Score'] = rfm[['R_score','F_score','M_score']].astype(int).sum(axis=1)

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

    # =========================
    # AUTO SELECT BEST K
    # =========================
    st.subheader("🔹 Silhouette Scores")

    scores = {}
    best_k = 2
    best_score = -1

    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, labels)
        scores[k] = score

        if score > best_score:
            best_score = score
            best_k = k

    st.write(scores)
    st.success(f"✅ Best K selected: {best_k}")

    # =========================
    # FINAL MODEL
    # =========================
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # =========================
    # SEGMENT NAMING (IMPROVED)
    # =========================
    def segment_customer(row):
        if row['RFM_Score'] >= 13:
            return "🌟 Champions"
        elif row['RFM_Score'] >= 10:
            return "💎 Loyal Customers"
        elif row['RFM_Score'] >= 7:
            return "🙂 Potential Loyalists"
        elif row['RFM_Score'] >= 5:
            return "⚠️ At Risk"
        else:
            return "❌ Lost Customers"

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)

    # =========================
    # METRICS
    # =========================
    st.subheader("📌 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(rfm))
    col2.metric("Average Revenue", round(rfm['Monetary'].mean(),2))
    col3.metric("Best K", best_k)

    # =========================
    # VISUALIZATIONS
    # =========================
    st.subheader("📊 Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Cluster Distribution")
        st.bar_chart(rfm['Cluster'].value_counts())

    with col2:
        st.write("Segment Distribution")
        st.bar_chart(rfm['Segment'].value_counts())

    st.write("RFM Score Distribution")
    st.bar_chart(rfm['RFM_Score'].value_counts())

    # =========================
    # FINAL OUTPUT
    # =========================
    st.subheader("🔹 Final Segmented Data")
    st.dataframe(rfm.head())

    # =========================
    # DOWNLOAD
    # =========================
    csv = rfm.to_csv().encode('utf-8')
    st.download_button("📥 Download Segmented Data", csv, "customer_segments.csv")

else:
    st.info("Please upload a dataset to begin.")
