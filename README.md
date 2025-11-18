# 1 - ML imports

## Common for all
```
Common Utilities
│
├── sklearn.model_selection
│   ├── train_test_split
│
├── sklearn.preprocessing
│   ├── StandardScaler
│   ├── OneHotEncoder
|
├── sklearn.metrics
│   ├── accuracy_score
│   ├── confusion_matrix
│   └── roc_auc_score
|
├── numpy
├── pandas 
|
```

## ML Models
```
Machine Learning Models
│
├── sklearn.linear_model
│   ├── LinearRegression
│   ├── LogisticRegression
│
├── sklearn.tree
│   ├── DecisionTreeClassifier
│
├── sklearn.ensemble
│   ├── RandomForestClassifier
│
├── sklearn.naive_bayes
│   ├── GaussianNB
│
├── sklearn.cluster
│   ├── KMeans
│   ├── DBSCAN // Density Based Clustering
│   └── AgglomerativeClustering // Hierarchical Clustering
│
├── sklearn.neighbors
│   ├── KNeighborsClassifier // K-Nearest Neighbors (Classification)
│   └── KNeighborsRegressor  // K-Nearest Neighbors (Regression)

```

# 2 - Data
## Load Data using pandas
```
import pandas as pd

# Load CSV file
df = pd.read_csv('path_to_your_file.csv')

# Display the first few rows
print(df.head())

```
Here are the names of the functions for loading datasets in different formats:

1. **CSV**: `pd.read_csv()`
2. **JSON**: `pd.read_json()`
3. **Excel**: `pd.read_excel()`
4. **Parquet**: `pd.read_parquet()`



