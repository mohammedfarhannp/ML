# Import Modules
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import pandas as pd

import matplotlib.pyplot as plt

# Load Data
File = r"..\DataSet\Gold Price Prediction.csv"
df = pd.read_csv(File)

# Handle Duplicates and Missing Values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].map(lambda x: x.toordinal())

# Split Features | Target
Target = 'Price Tomorrow'

x = df.drop(Target, axis=1)
y = df[Target]

# Create Train and Test Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Creation and Training
Model = LinearRegression()
Model.fit(x_train, y_train)

# Test Prediction
Y_Pred = Model.predict(x_test)

# Evaluation
print("\n===== Evaluation =====")
print(f"Mean Squared Error: {mean_squared_error(y_test, Y_Pred)}")
print(f"R Square : {r2_score(y_test, Y_Pred)}")

# Plot Evaluation
plt.scatter(y_test, Y_Pred)
plt.xlabel("Actual Gold Prices")
plt.ylabel("Predicted Gold Prices")
plt.title("Actual vs Predicted Gold Price")
plt.show()

# New Data
new_data = pd.DataFrame([{
    "Date": pd.to_datetime("2025-01-15").toordinal(),
    "Price 2 Days Prior": 2100,
    "Price 1 Day Prior": 2120,
    "Price Today": 2110,
    "Price Change Tomorrow": 5,
    "Price Change Ten": 12,
    "Std Dev 10": 15.4,
    "Twenty Moving Average": 2080,
    "Fifty Day Moving Average": 2055,
    "200 Day Moving Average": 1990,
    "Monthly Inflation Rate": 3.2,
    "EFFR Rate": 5.3,
    "Volume ": 150000,
    "Treasury Par Yield Month": 4.1,
    "Treasury Par Yield Two Year": 4.7,
    "Treasury Par Yield Curve Rates (10 Yr)": 4.3,
    "DXY": 103.2,
    "SP Open": 4700,
    "VIX": 15.2,
    "Crude": 78.5
}])

pred = Model.predict(new_data)[0]
print("\nNew Data\nPredicted Price Tomorrow:", pred)

