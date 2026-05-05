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
# REQUIRED FORMAT DISPLAY
# =========================
st.subheader("📋 Required Dataset Format")

st.markdown("""
Your dataset must contain:
- CustomerID
- InvoiceNo
- InvoiceDate
- Quantity
- UnitPrice
""")

# =========================
# SAMPLE DATA
# =========================
@st.cache_data
def create_sample_data():
    data = {
        "CustomerID": np.random.choice(range(10000, 10050), 200),
        "InvoiceNo": np.random.choice([f"INV{i}" for i in range(1000,1100)], 200),
        "InvoiceDate": pd.date_range(start="2024-01-01", periods=200, freq="D"),
        "Quantity": np.random.randint(1, 10, 200),
        "UnitPrice": np.random.uniform(50, 500, 200).round(2)
    }
    return pd.DataFrame(data)

sample_df = create_sample_data()
st.dataframe(sample_df.head())

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])

if file is not None:

    df = pd.read_csv(file, encoding='ISO-8859-1') if file.name.endswith('.csv') else pd.read_excel(file)

    # =========================
    # VALIDATION
    # =========================
    required_columns = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']
    if not all(col in df.columns for col in required_columns):
        st.error("Missing required columns!")
        st.stop()

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

    # =========================
    # REMOVE OUTLIERS
    # =========================
    rfm = rfm[
        (rfm['Monetary'] < rfm['Monetary'].quantile(0.99)) &
        (rfm['Frequency'] < rfm['Frequency'].quantile(0.99))
    ]

    st.subheader("🔹 RFM Table")
    st.dataframe(rfm.head())

    # =========================
    # RFM SCORING
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
    # FIND BEST K
    # =========================
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

    st.write("Silhouette Scores:", scores)
    st.success(f"Best K: {best_k}")

    # =========================
    # FINAL MODEL
    # =========================
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # =========================
    # FINAL SEGMENTATION (YOUR REQUIREMENT)
    # =========================
    def segment_customer(row):

        if row['RFM_Score'] >= 13:
            return "Loyal High-Spenders"

        elif row['RFM_Score'] >= 9:
            return "Regular Customers"

        elif row['RFM_Score'] >= 5:
            return "New/Occasional Customers"

        else:
            return "At-Risk Infrequents"

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)

    # =========================
    # METRICS
    # =========================
    st.subheader("📌 Key Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(rfm))
    col2.metric("Avg Revenue", round(rfm['Monetary'].mean(),2))
    col3.metric("Best K", best_k)

    # =========================
    # VISUALS
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
    st.download_button("Download CSV", csv, "customer_segments.csv")

else:
    st.info("Upload a dataset to start")
