# Import Modules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import pandas as pd

# Load Data
File = r"..\DataSet\wine-clustering.csv"
df = pd.read_csv(File)

# Handle Duplicates and Missing Values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Select Features (All numerical columns)
x = df.select_dtypes(include='number')

# Model Creation and Training
Model = KMeans(n_clusters=3, random_state=42)
Model.fit(x)

# Cluster Labels
labels = Model.labels_

# Evaluation Metrics (Unsupervised)
print("\n===== K-Means Evaluation =====")
print(f"Inertia (WCSS): {Model.inertia_}")
print(f"Silhouette Score: {silhouette_score(x, labels)}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(x, labels)}")

# Attach clusters to dataframe (optional)
df['Cluster'] = labels

# Show sample clustered data
print("\nClustered Data (first 5 rows):")
print(df.head())

# ================================
#   NEW DATA PREDICTION
# ================================
# Example new observation (must match feature columns)
new_data = pd.DataFrame([{
    col: df[col].mean()   # you can replace with real user input values
    for col in x.columns
}])

# Predict cluster
new_cluster = Model.predict(new_data)[0]

print("\n===== New Data Prediction =====")
print("New Data:\n", new_data)
print(f"\nPredicted Cluster: {new_cluster}")
