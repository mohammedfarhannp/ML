# Import Modules
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

# Load Data
File = r"..\DataSet\wine-clustering.csv"
df = pd.read_csv(File)

# Handle Duplicates and Missing Values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Select Features (All numerical columns)
x = df.select_dtypes(include='number')

# Scaling (Agglomerative works better with scaling)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Model Creation and Training
Model = AgglomerativeClustering(n_clusters=3)
labels = Model.fit_predict(x_scaled)

# Evaluation Metrics
print("\n===== Agglomerative Clustering Evaluation =====")
print(f"Silhouette Score: {silhouette_score(x_scaled, labels)}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(x_scaled, labels)}")

# Attach clusters to dataframe (optional)
df["Cluster"] = labels

# Show sample data with clusters
print("\nClustered Data (first 5 rows):")
print(df.head())

#   NEW DATA PREDICTION (manual, since no .predict())
# Step 1: Compute cluster centers manually
cluster_centers = []

for c in sorted(df["Cluster"].unique()):
    center = x_scaled[df["Cluster"] == c].mean(axis=0)
    cluster_centers.append(center)

cluster_centers = np.array(cluster_centers)

# Step 2: New Data Input
new_data = pd.DataFrame([{
    col: df[col].mean()   # replace with real input values
    for col in x.columns
}])

# Scale new data
new_scaled = scaler.transform(new_data)

# Step 3: Assign to nearest cluster center
distances = np.linalg.norm(new_scaled - cluster_centers, axis=1)
pred_cluster = np.argmin(distances)

print("\n===== New Data Prediction =====")
print("New Data:\n", new_data)
print(f"\nPredicted Cluster: {pred_cluster}")
