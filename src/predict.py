import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD SAVED MODEL
# =========================

kmeans = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

print("K-Means model loaded successfully.")
print("Scaler loaded successfully.")

# =========================
# CLUSTERING FEATURES
# =========================

clustering_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "TotalQuantity"
]

# =========================
# SAMPLE CUSTOMER
# =========================

customer = pd.DataFrame([{
    "Recency": 10,
    "Frequency": 8,
    "Monetary": 5000,
    "AvgOrderValue": 625,
    "UniqueProducts": 100,
    "TotalQuantity": 2500
}])

print("\n========== CUSTOMER INPUT ==========")
print(customer)

# =========================
# LOG TRANSFORMATION
# =========================

customer_log = customer.copy()

for column in clustering_features:
    customer_log[column] = np.log1p(customer_log[column])

print("\n========== LOG TRANSFORMED INPUT ==========")
print(customer_log)

# =========================
# SCALE FEATURES
# =========================

customer_scaled = pd.DataFrame(
    scaler.transform(customer_log),
    columns=clustering_features
)

print("\n========== SCALED INPUT ==========")
print(customer_scaled)

# =========================
# PREDICT CLUSTER
# =========================

cluster = kmeans.predict(customer_scaled)[0]

print("\n========== PREDICTION ==========")
print(f"Predicted Customer Cluster: {cluster}")

# =========================
# CUSTOMER SEGMENT LABEL
# =========================

segment_names = {
    0: "Moderate Engagement / Moderate Value",
    1: "High-Value / Low-Frequency",
    2: "Highly Engaged / High-Value",
    3: "Inactive / Low-Value"
}

segment = segment_names[cluster]

print(f"Customer Segment: {segment}")