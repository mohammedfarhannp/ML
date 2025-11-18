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

## Preprocess Data (Clean up)
### step 1 - Drop Duplicates
```
# Remove duplicate rows
df.drop_duplicates(inplace=True)
```

### step 2 - Handle Missing Values
#### Either Delete missing values
```
# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (if necessary)
df.dropna(inplace=True)
```

#### Or fill missing values

```
# Alternatively, fill missing values (e.g., with the mean or median)
df.fillna(df.mean(), inplace=True)  # for numerical columns
```
### step 3 - Feature Engineering
#### OneHotEncoder for Categorical Features
```
from sklearn.preprocessing import OneHotEncoder

# Apply One-Hot Encoding to categorical columns (e.g., 'category_column')
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_data = encoder.fit_transform(df[['category_column']])

# Convert encoded data back to a DataFrame and add to the original dataframe
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['category_column']))
df = pd.concat([df, encoded_df], axis=1).drop(['category_column'], axis=1)
```

#### StandardScaler for scaled features (logistic regression, KNN)
```
from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to features (X), but exclude the target column
X_scaled = scaler.fit_transform(df.drop('target_column', axis=1))
```

### step 4 - Seperate Features and Target
```
x = df.drop('target_column', axis=1)  # Features
y = df['target_column']               # Target
```

## Create Train and Test set
```
from sklearn.model_selection import train_test_split

# Split data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```
