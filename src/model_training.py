import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# Create results directory
os.makedirs("results", exist_ok=True)


# =========================
# LOAD CUSTOMER DATA
# =========================

customer_features = pd.read_csv("data/customer_features.csv")

print("\n========== CUSTOMER DATA ==========")

print("\nDataset shape:")
print(customer_features.shape)

print("\nFirst 5 rows:")
print(customer_features.head())

print("\nColumns:")
print(customer_features.columns.tolist())

# =========================
# SELECT CLUSTERING FEATURES
# =========================

clustering_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "TotalQuantity"
]

X = customer_features[clustering_features].copy()

print("\n========== CLUSTERING FEATURES ==========")

print("\nSelected features:")
print(clustering_features)

print("\nFeature data shape:")
print(X.shape)

print("\nFirst 5 rows:")
print(X.head())

# =========================
# LOG TRANSFORMATION
# =========================

X_log = X.copy()

for column in clustering_features:
    X_log[column] = np.log1p(X_log[column])

print("\n========== LOG TRANSFORMED FEATURES ==========")

print("\nFirst 5 rows:")
print(X_log.head())

print("\nStatistics after log transformation:")
print(X_log.describe())

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=clustering_features
)

print("\n========== SCALED FEATURES ==========")

print("\nFirst 5 rows:")
print(X_scaled.head())

print("\nScaled feature statistics:")
print(X_scaled.describe())

# =========================
# ELBOW METHOD
# =========================

print("\n========== ELBOW METHOD ==========")

inertia = []
k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

    print(f"K={k}, Inertia={kmeans.inertia_:.2f}")


# Plot Elbow Curve
plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")

plt.xticks(list(k_values))

plt.tight_layout()

plt.savefig("results/elbow_curve.png")

plt.close()

print("\nElbow curve saved to results/elbow_curve.png")

# =========================
# SILHOUETTE SCORE
# =========================

print("\n========== SILHOUETTE SCORES ==========")

silhouette_scores = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(f"K={k}, Silhouette Score={score:.4f}")


# Plot Silhouette Scores
plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Score for Different K Values")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.xticks(list(k_values))

plt.tight_layout()

plt.savefig("results/silhouette_scores.png")

plt.close()

print("\nSilhouette score plot saved to results/silhouette_scores.png")

# =========================
# COMPARE K-MEANS CLUSTERS
# =========================

print("\n========== K-MEANS CLUSTER COMPARISON ==========")

for k in [2, 3, 4]:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    customer_features[f"Cluster_{k}"] = labels

    print(f"\nK={k}")
    print("Cluster sizes:")
    print(
        customer_features[f"Cluster_{k}"]
        .value_counts()
        .sort_index()
    )
    
    # =========================
# CLUSTER PROFILING
# =========================

print("\n========== CLUSTER PROFILES ==========")

profile_features = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "TotalQuantity"
]

for k in [2, 3, 4]:

    print(f"\n----- K={k} CLUSTER PROFILE -----")

    profile = customer_features.groupby(
        f"Cluster_{k}"
    )[profile_features].mean().round(2)

    print(profile)
    
    # =========================
# PCA VISUALIZATION
# =========================

from sklearn.decomposition import PCA

print("\n========== PCA VISUALIZATION ==========")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

print("Total explained variance:")
print(pca.explained_variance_ratio_.sum())


# Create PCA dataframe
pca_data = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)


# Visualize K=4 clusters
plt.figure(figsize=(10, 7))

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue=customer_features["Cluster_4"],
    palette="viridis",
    data=pca_data,
    alpha=0.6
)

plt.title("Customer Segments - K-Means (K=4)")
plt.xlabel(
    f"Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
)

plt.ylabel(
    f"Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
)

plt.tight_layout()

plt.savefig("results/pca_k4_clusters.png")
dpi=300
plt.close()

print("\nPCA cluster visualization saved to results/pca_k4_clusters.png")

# =========================
# CLUSTER PROFILE HEATMAP
# =========================

print("\n========== CLUSTER PROFILE HEATMAP ==========")

# Use K=4 for profile visualization
cluster_profile = customer_features.groupby(
    "Cluster_4"
)[profile_features].mean()

# Standardize profile values for visualization
profile_scaled = (
    cluster_profile - cluster_profile.mean()
) / cluster_profile.std()

plt.figure(figsize=(12, 6))

sns.heatmap(
    profile_scaled,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("K-Means Customer Cluster Profiles (K=4)")
plt.xlabel("Customer Features")
plt.ylabel("Cluster")

plt.tight_layout()

plt.savefig("results/cluster_profile_heatmap.png")

plt.close()

print("\nCluster profile heatmap saved to results/cluster_profile_heatmap.png")

# =========================
# FINAL K-MEANS COMPARISON
# =========================

print("\n========== FINAL K-MEANS COMPARISON ==========")

for k in [2, 3, 4]:

    print(f"\nK={k}")

    comparison = customer_features.groupby(
        f"Cluster_{k}"
    )[profile_features].mean().round(2)

    print(comparison)
    
    # =========================
# FINAL K-MEANS MODEL
# =========================

print("\n========== FINAL K-MEANS MODEL ==========")

final_k = 4

final_kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10
)

customer_features["Cluster"] = final_kmeans.fit_predict(X_scaled)

print("\nFinal cluster sizes:")
print(
    customer_features["Cluster"]
    .value_counts()
    .sort_index()
)


# =========================
# SAVE MODEL
# =========================

os.makedirs("models", exist_ok=True)

import joblib

joblib.dump(
    final_kmeans,
    "models/kmeans_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nK-Means model saved to models/kmeans_model.pkl")
print("Scaler saved to models/scaler.pkl")


# =========================
# SAVE SEGMENTED CUSTOMERS
# =========================

customer_features.to_csv(
    "results/customer_segments.csv",
    index=False
)

print("Customer segments saved to results/customer_segments.csv")

# =========================
# HIERARCHICAL CLUSTERING
# =========================

print("\n========== HIERARCHICAL CLUSTERING ==========")

linked = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(12, 7))

dendrogram(
    linked,
    truncate_mode="lastp",
    p=30
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Groups")
plt.ylabel("Distance")

plt.tight_layout()
plt.savefig("results/hierarchical_dendrogram.png")
plt.close()

print("\nDendrogram saved to results/hierarchical_dendrogram.png")

# =========================
# AGGLOMERATIVE CLUSTERING
# =========================

print("\n========== AGGLOMERATIVE CLUSTERING ==========")

hierarchical_scores = []

for k in [2, 3, 4]:
    hierarchical_model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = hierarchical_model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    hierarchical_scores.append(score)

    print(f"K={k}, Silhouette Score={score:.4f}")

    print("Cluster sizes:")
    print(
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )
    
    # =========================
# HIERARCHICAL CLUSTER PROFILES
# =========================

print("\n========== HIERARCHICAL CLUSTER PROFILES ==========")

for k in [2, 3, 4]:

    hierarchical_model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = hierarchical_model.fit_predict(X_scaled)

    customer_features[f"Hierarchical_{k}"] = labels

    profile = customer_features.groupby(
        f"Hierarchical_{k}"
    )[profile_features].mean().round(2)

    print(f"\n----- HIERARCHICAL K={k} -----")
    print(profile)
    
    # =========================
# HIERARCHICAL PCA VISUALIZATION
# =========================

print("\n========== HIERARCHICAL PCA VISUALIZATION ==========")

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue=customer_features["Hierarchical_4"],
    palette="viridis",
    data=pca_data,
    alpha=0.6
)

plt.title("Hierarchical Clustering - K=4")
plt.xlabel(
    f"Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
)

plt.ylabel(
    f"Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
)

plt.tight_layout()

plt.savefig(
    "results/pca_hierarchical_k4_clusters.png",
    dpi=300
)

plt.close()

print(
    "\nHierarchical PCA visualization saved to "
    "results/pca_hierarchical_k4_clusters.png"
)

# =========================
# DBSCAN CLUSTERING
# =========================

print("\n========== DBSCAN CLUSTERING ==========")

dbscan_results = []

eps_values = [0.5, 0.7, 0.9, 1.1]
min_samples = 10

for eps in eps_values:

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(
        set(labels) - {-1}
    )

    n_noise = list(labels).count(-1)

    print(
        f"eps={eps}, "
        f"Clusters={n_clusters}, "
        f"Noise Points={n_noise}"
    )

    # Silhouette score requires at least 2 clusters
    non_noise_mask = labels != -1

    if (
        n_clusters >= 2
        and non_noise_mask.sum() > n_clusters
    ):
        score = silhouette_score(
            X_scaled[non_noise_mask],
            labels[non_noise_mask]
        )
        print(f"Silhouette Score={score:.4f}")
    else:
        score = None
        print("Silhouette Score=Not available")

    dbscan_results.append(
        {
            "eps": eps,
            "min_samples": min_samples,
            "clusters": n_clusters,
            "noise_points": n_noise,
            "silhouette_score": score
        }
    )
    
    # =========================
# FINAL DBSCAN MODEL
# =========================

print("\n========== FINAL DBSCAN MODEL ==========")

final_dbscan = DBSCAN(
    eps=0.7,
    min_samples=10
)

dbscan_labels = final_dbscan.fit_predict(X_scaled)

customer_features["DBSCAN_Cluster"] = dbscan_labels

print("\nDBSCAN cluster distribution:")
print(
    customer_features["DBSCAN_Cluster"]
    .value_counts()
    .sort_index()
)

print(
    f"\nNumber of clusters: "
    f"{len(set(dbscan_labels) - {-1})}"
)

print(
    f"Number of noise points: "
    f"{list(dbscan_labels).count(-1)}"
)
# =========================
# SAVE DBSCAN RESULTS
# =========================

customer_features.to_csv(
    "results/customer_segments.csv",
    index=False
)

print(
    "\nDBSCAN results added to "
    "results/customer_segments.csv"
)

# =========================
# DBSCAN CLUSTER PROFILES
# =========================

print("\n========== DBSCAN CLUSTER PROFILES ==========")

dbscan_profile = customer_features.groupby(
    "DBSCAN_Cluster"
)[profile_features].mean().round(2)

print(dbscan_profile)

# =========================
# DBSCAN NOISE ANALYSIS
# =========================

print("\n========== DBSCAN NOISE ANALYSIS ==========")

noise_customers = customer_features[
    customer_features["DBSCAN_Cluster"] == -1
]

print("Number of noise customers:", len(noise_customers))

print("\nNoise customer profile:")

print(
    noise_customers[profile_features]
    .mean()
    .round(2)
)

# ============================================================
# ISOLATION FOREST ANOMALY DETECTION
# ============================================================

print("\n========== ISOLATION FOREST ==========")

isolation_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

anomaly_labels = isolation_forest.fit_predict(X_scaled)

customer_features["Anomaly"] = anomaly_labels

# Convert labels:
# 1  = Normal
# -1 = Anomaly
n_anomalies = (anomaly_labels == -1).sum()
n_normal = (anomaly_labels == 1).sum()

print(f"Normal customers: {n_normal}")
print(f"Anomalous customers: {n_anomalies}")

# Anomaly percentage
anomaly_percentage = (n_anomalies / len(customer_features)) * 100

print(f"Anomaly percentage: {anomaly_percentage:.2f}%")

# Save anomaly results
customer_features.to_csv(
    "results/customer_segments.csv",
    index=False
)

print("Isolation Forest results added to results/customer_segments.csv")

# ============================================================
# ISOLATION FOREST ANOMALY PROFILE
# ============================================================

print("\n========== ISOLATION FOREST ANOMALY PROFILE ==========")

anomaly_profile = customer_features[
    customer_features["Anomaly"] == -1
][clustering_features].mean()

print("Average profile of anomalous customers:")
print(anomaly_profile)

normal_profile = customer_features[
    customer_features["Anomaly"] == 1
][clustering_features].mean()

print("\nAverage profile of normal customers:")
print(normal_profile)

# ============================================================
# TOP ANOMALOUS CUSTOMERS
# ============================================================

print("\n========== TOP ANOMALOUS CUSTOMERS ==========")

customer_features["AnomalyScore"] = (
    isolation_forest.decision_function(X_scaled)
)

top_anomalies = customer_features[
    customer_features["Anomaly"] == -1
].sort_values("AnomalyScore")

print(
    top_anomalies[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "AvgOrderValue",
            "UniqueProducts",
            "TotalQuantity",
            "AnomalyScore"
        ]
    ].head(10)
)