# Import Modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# Load Data
File = "..\\DataSet\\wine-clustering.csv"
df = pd.read_csv(File)

# Show Original Data
print("Original Data (First 5 rows):")
print(df.head())

# ==============================
# Standardization
# ==============================
scaler_standard = StandardScaler()

standard_data = scaler_standard.fit_transform(df)
standard_df = pd.DataFrame(standard_data, columns=df.columns)

print("\nStandardized Data (First 5 rows):")
print(standard_df.head())

# ==============================
# Normalization
# ==============================
scaler_normal = MinMaxScaler()

normal_data = scaler_normal.fit_transform(df)
normal_df = pd.DataFrame(normal_data, columns=df.columns)

print("\nNormalized Data (First 5 rows):")
print(normal_df.head())
