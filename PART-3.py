import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set plotting parameters
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=" * 60)
print("RUNNING PART 4: K-MEANS CLUSTERING PIPELINE")
print("=" * 60)

# -------------------------------------------------------------------------
# TASK 1: Load and Filter Features for Clustering
# -------------------------------------------------------------------------
print("\n--- Task 1: Filtering Features ---")
df = pd.read_csv("cleaned_data.csv")

# Select numerical features relevant to pricing and spatial footprints
clustering_features = ["Square_Footage", "Price", "Year_Built"]
X_clus = df[clustering_features].dropna()

print(f"Data subset selected for clustering shape: {X_clus.shape}")
print(X_clus.head())

# -------------------------------------------------------------------------
# TASK 2: Feature Standardization
# -------------------------------------------------------------------------
print("\n--- Task 2: Scaling Clustering Inputs ---")
scaler_clus = StandardScaler()
X_clus_scaled = scaler_clus.fit_transform(X_clus)

X_scaled_df = pd.DataFrame(X_clus_scaled, columns=clustering_features)

# -------------------------------------------------------------------------
# TASK 3: Hyperparameter Optimization (Elbow & Silhouette)
# -------------------------------------------------------------------------
print("\n--- Task 3: Hyperparameter Search ($K=2$ to $K=10$) ---")

wcss = []  # Within-Cluster Sum of Squares
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_clus_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_clus_scaled, kmeans.labels_))

# Plot 1: Elbow Method Curve
plt.figure()
plt.plot(k_range, wcss, marker="o", linestyle="--", color="b", linewidth=2)
plt.title("Elbow Method For Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("plot_11_elbow.png")
plt.close()

# Plot 2: Silhouette Score Plot
plt.figure()
plt.plot(k_range, silhouette_scores, marker="s", linestyle="-", color="g", linewidth=2)
plt.title("Silhouette Coefficients Across Cluster Ranges")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("plot_12_silhouette.png")
plt.close()

print("Optimization plots saved as 'plot_11_elbow.png' and 'plot_12_silhouette.png'.")

# Find optimal K automatically via highest silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Empirically determined optimal clusters based on max Silhouette Score: K = {optimal_k}")

# -------------------------------------------------------------------------
# TASK 4 & 5: Train Optimal K-Means & Append Cluster Allocations
# -------------------------------------------------------------------------
print(f"\n--- Tasks 4 & 5: Fit Optimal K-Means (K={optimal_k}) ---")
final_kmeans = KMeans(n_clusters=optimal_k, init="k-means++", random_state=42, n_init=10)
df["Cluster_Labels"] = final_kmeans.fit_predict(X_clus_scaled)

print("\nValue distributions across generated labels:")
print(df["Cluster_Labels"].value_counts())

# -------------------------------------------------------------------------
# TASK 6: Segmentation Visualization
# -------------------------------------------------------------------------
print("\n--- Task 6: Visualizing Spatial Segmentation ---")

# Scatter plot: Square Footage vs Price colored by Cluster Assignment
plt.figure()
sns.scatterplot(
    data=df,
    x="Square_Footage",
    y="Price",
    hue="Cluster_Labels",
    palette="viridis",
    style="Cluster_Labels",
    s=80,
    alpha=0.8,
)
plt.title(f"Property Segment Subspaces ($K={optimal_k}$ Segments)")
plt.xlabel("Square Footage")
plt.ylabel("Price ($)")
plt.legend(title="Cluster Group")
plt.tight_layout()
plt.savefig("plot_13_clusters_scatter.png")
plt.close()

print("Clustering scatter matrix saved as 'plot_13_clusters_scatter.png'.")

# -------------------------------------------------------------------------
# TASK 7: Cluster Profiling & Aggregation
# -------------------------------------------------------------------------
print("\n--- Task 7: Segment Summarization & Profiling ---")
profile_metrics = df.groupby("Cluster_Labels")[clustering_features].mean().reset_index()

print("\n--- Cluster Profile Summary (Mean Metrics per Group) ---")
print(profile_metrics)

print("\n" + "=" * 60)
print("ENTIRE ASSIGNMENT WORKFLOW COMPLETELY FINISHED!")
print("=" * 60)