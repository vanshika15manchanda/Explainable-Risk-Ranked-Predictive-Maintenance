import pandas as pd
from IPython.display import display

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

print("Loading dataset...\n")
df = pd.read_csv(url)

print("--- 1. DATASET SHAPE ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}\n")

print("--- 2. DATASET INFO ---")
df.info()
print("\n")

print("--- 3. SUMMARY STATISTICS ---")

display(df.describe())


print("\n--- 4. COLUMN VERIFICATION ---")
print("Here are all the columns currently in the dataset:\n", df.columns.tolist(), "\n")

# verifying the specific columns needed for the project
expected_columns = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Machine failure',
    'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]

print("Checking for your target features and failure types:")
actual_columns = df.columns.tolist()
for col in expected_columns:
    if col in actual_columns:
        print(f"Found: '{col}'")
    else:
        print(f" Missing: '{col}' (Check for typos or formatting differences)")
