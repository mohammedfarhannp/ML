# Import Modules
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import pandas as pd

# Load Data
File = "..\\DataSet\\Social_Network_Ads.csv"
df = pd.read_csv(File)

print(df.head())

# Select Features and Target
# Assuming columns: Age, EstimatedSalary, Purchased
x = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Data Scaling
sc = StandardScaler()
x_scaled = sc.fit_transform(x)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.20, random_state=42
)

# Model
model = SVC(kernel='rbf')     # rbf is best in most cases

# Train Model
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Model Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------
# New Data Prediction
# ------------------------------------------------

# New Data (example)
new_data = {
    'Age': [35],
    'EstimatedSalary': [55000]
}

new_df = pd.DataFrame(new_data)

# Scale New Data
new_scaled = sc.transform(new_df)

# Prediction
new_pred = model.predict(new_scaled)
print("\nNew Prediction:", new_pred)
