# Import Modules
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import pandas as pd

# Load Data
File = r"..\DataSet\diabetes.csv"
df = pd.read_csv(File)

# Handle Duplicates and Missing Values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Split Features | Target
Target = 'Outcome'

x = df.drop(Target, axis=1)
y = df[Target]

# Create Test and Train Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Creation and Training
Model = LogisticRegression(max_iter=1000)
Model.fit(x_train, y_train)

# Test Prediction
Y_Pred = Model.predict(x_test)

# Evaluation
print(f"\nAccuracy of Model: {accuracy_score(y_test, Y_Pred)}\n")
print(f"\nConfusion Matrix\n{confusion_matrix(y_test, Y_Pred)}\n")
print(f"\nClassification Report\n{classification_report(y_test, Y_Pred)}\n")

# New Data Test
new_data = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 32
}])

# Predict class (0 = No Diabetes, 1 = Diabetes)
prediction = Model.predict(new_data)[0]

# Predict probability (how confident the model is)
probability = Model.predict_proba(new_data)[0][1]   # Probability of class 1

print("\n===== New Prediction =====")
print(f"Predicted Outcome : {prediction}")
print(f"Probability of Diabetes : {probability:.4f}")