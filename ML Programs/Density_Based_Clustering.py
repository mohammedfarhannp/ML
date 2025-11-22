# Import Modules
from sklearn.cluster import DBSCAN
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

# Scale Data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Model Creation and Training
Model = DBSCAN(eps=2.0, min_samples=3)
labels = Model.fit_predict(x_scaled)

df["Cluster"] = labels

# Evaluation Metrics (only if at least 2 clusters)
if len(set(labels)) > 1 and -1 not in set(labels):
    print("\n===== DBSCAN Evaluation =====")
    print(f"Silhouette Score: {silhouette_score(x_scaled, labels)}")
    print(f"Davies-Bouldin Score: {davies_bouldin_score(x_scaled, labels)}")
else:
    print("\n===== DBSCAN Evaluation =====")
    print("Cannot compute Silhouette or Davies-Bouldin (clusters = 1 or noise present).")

# Show results
print("\nClustered Data (first 5 rows):")
print(df.head())

# Identify core samples
core_mask = Model.core_sample_indices_

if len(core_mask) == 0:
    print("\n===== New Data Prediction =====")
    print("DBSCAN found NO CORE POINTS — cannot predict new data.")
else:
    core_points = x_scaled[core_mask]
    core_labels = labels[core_mask]

    # New data example
    new_data = pd.DataFrame([{
        col: df[col].mean()
        for col in x.columns
    }])

    # Scale new data
    new_scaled = scaler.transform(new_data)

    # Compute distance to all core points
    distances = np.linalg.norm(core_points - new_scaled, axis=1)

    nearest_index = np.argmin(distances)
    pred_cluster = core_labels[nearest_index]

    print("\n===== New Data Prediction =====")
    print("New Data:\n", new_data)
    print(f"\nPredicted Cluster: {pred_cluster}")