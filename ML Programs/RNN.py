# Import Modules
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load Data
File = r"..\DataSet\Gold Price Prediction.csv"
df = pd.read_csv(File)

# Sort by Date (very important for RNN)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Use only the gold price column (Rename if needed)
target_col = 'Price Today'     # <<< CHANGE IF YOUR COLUMN NAME IS DIFFERENT
data = df[[target_col]].values

# Scale Values
Scaler = MinMaxScaler()
scaled_data = Scaler.fit_transform(data)

# Create Time-Series Sequences
def create_sequences(dataset, step):
    x_list = []
    y_list = []
    for i in range(step, len(dataset)):
        x_list.append(dataset[i-step:i, 0])
        y_list.append(dataset[i, 0])
    return np.array(x_list), np.array(y_list)

time_steps = 10
x, y = create_sequences(scaled_data, time_steps)

# Reshape for RNN input
x = x.reshape(x.shape[0], x.shape[1], 1)

# Split Train | Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# Model Creation and Training
Model = Sequential()
Model.add(SimpleRNN(64, activation='tanh', return_sequences=False))
Model.add(Dense(1))

Model.compile(optimizer='adam', loss='mse')
Model.fit(x_train, y_train, epochs=20, batch_size=32)

# Test Prediction
y_pred = Model.predict(x_test)

# Inverse scaling
y_test_real = Scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_real = Scaler.inverse_transform(y_pred)

# Display Last Predictions
print("\nActual :", y_test_real[-5:].flatten())
print("Predicted :", y_pred_real[-5:].flatten())

# New Data Prediction
# ---------------------------------------------
# You must provide last 10 days’ prices:
new_data_input = np.array([
    57750, 57820, 57910, 58000, 58100,
    58220, 58310, 58400, 58550, 58600
]).reshape(-1, 1)

new_scaled = Scaler.transform(new_data_input)

new_scaled = new_scaled.reshape(1, time_steps, 1)

new_pred = Model.predict(new_scaled)[0][0]
new_pred_real = Scaler.inverse_transform([[new_pred]])[0][0]

print("\n===== New Prediction =====")
print(f"Predicted Gold Price Tomorrow : {new_pred_real}")
