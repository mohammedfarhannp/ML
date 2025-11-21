# Import Module
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# Load Data
File = r"..\DataSet\breast_cancer.csv"
df = pd.read_csv(File)

# Handle Duplicates and Missing Values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

Target = 'target'

# Convert diagnosis to numeric (M=1, B=0) if needed
if df[Target].dtype == 'object':
    df[Target] = df[Target].map({'M': 1, 'B': 0})

# Split Features | Target
x = df.drop(Target, axis=1)
y = df[Target]

# Create Test and Train Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Creation and Training
Model = RandomForestClassifier()
Model.fit(x_train, y_train)

# Test Prediction
Y_Pred = Model.predict(x_test)

# Evaluation
print("\n===== Evaluation =====")
print("Accuracy :", accuracy_score(y_test, Y_Pred))
print("\nClassification Report:\n", classification_report(y_test, Y_Pred))

# Confusion Matrix
cm = confusion_matrix(y_test, Y_Pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()