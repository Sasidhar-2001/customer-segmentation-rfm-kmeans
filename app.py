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
# REQUIRED FORMAT
# =========================
st.subheader("📋 Required Dataset Format")
st.markdown("""
- CustomerID → Unique ID  
- InvoiceNo → Transaction ID  
- InvoiceDate → Date  
- Quantity → Items purchased  
- UnitPrice → Price per item  
""")

# =========================
# SAMPLE DATA
# =========================
st.subheader("📥 Sample Dataset")

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

sample_csv = sample_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Sample Dataset", sample_csv, "sample_data.csv")

# =========================
# LOAD FUNCTION
# =========================
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, encoding='ISO-8859-1')
    else:
        return pd.read_excel(file)

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if file is not None:

    df = load_data(file)

    # =========================
    # VALIDATION
    # =========================
    required_cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice']
    if any(col not in df.columns for col in required_cols):
        st.error("❌ Missing required columns")
        st.stop()

    # Optional raw preview
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.head())

    # =========================
    # PREPROCESSING
    # =========================
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # =========================
    # RFM
    # =========================
    snapshot = df['InvoiceDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot - x.max()).days,
        'InvoiceNo': 'count',
        'TotalPrice': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    rfm = rfm[
        (rfm['Monetary'] < rfm['Monetary'].quantile(0.99)) &
        (rfm['Frequency'] < rfm['Frequency'].quantile(0.99))
    ]

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
    # BEST K
    # =========================
    scores = {}
    best_k, best_score = 2, -1

    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, labels)
        scores[k] = score

        if score > best_score:
            best_k, best_score = k, score

    st.write("Silhouette Scores:", scores)
    st.success(f"Best K: {best_k}")

    # =========================
    # FINAL MODEL
    # =========================
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # =========================
    # SEGMENTATION
    # =========================
    def segment(row):
        if row['RFM_Score'] >= 13:
            return "Champions"
        elif row['RFM_Score'] >= 10:
            return "Loyal"
        elif row['RFM_Score'] >= 7:
            return "Potential"
        elif row['RFM_Score'] >= 5:
            return "At Risk"
        else:
            return "Lost"

    rfm['Segment'] = rfm.apply(segment, axis=1)

    # =========================
    # SIDEBAR FILTERS
    # =========================
    st.sidebar.header("🔍 Filters")

    seg_filter = st.sidebar.multiselect(
        "Segment",
        rfm['Segment'].unique(),
        default=rfm['Segment'].unique()
    )

    cluster_filter = st.sidebar.multiselect(
        "Cluster",
        sorted(rfm['Cluster'].unique()),
        default=sorted(rfm['Cluster'].unique())
    )

    score_range = st.sidebar.slider(
        "RFM Score",
        int(rfm['RFM_Score'].min()),
        int(rfm['RFM_Score'].max()),
        (int(rfm['RFM_Score'].min()), int(rfm['RFM_Score'].max()))
    )

    # Apply filters
    filtered = rfm[
        (rfm['Segment'].isin(seg_filter)) &
        (rfm['Cluster'].isin(cluster_filter)) &
        (rfm['RFM_Score'] >= score_range[0]) &
        (rfm['RFM_Score'] <= score_range[1])
    ]

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", len(filtered))
    col2.metric("Avg Revenue", round(filtered['Monetary'].mean(),2))
    col3.metric("Clusters", best_k)

    # =========================
    # VISUALS
    # =========================
    st.subheader("📊 Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(filtered['Cluster'].value_counts())

    with col2:
        st.bar_chart(filtered['Segment'].value_counts())

    st.bar_chart(filtered['RFM_Score'].value_counts())

    # =========================
    # OUTPUT
    # =========================
    st.subheader("🔹 Filtered Data")
    st.dataframe(filtered.head())

    # =========================
    # DOWNLOAD
    # =========================
    csv = filtered.to_csv().encode('utf-8')
    st.download_button("📥 Download Filtered Data", csv, "filtered_customers.csv")

else:
    st.info("Upload a dataset to start.")
