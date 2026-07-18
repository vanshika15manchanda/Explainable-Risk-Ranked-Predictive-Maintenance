import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

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


print('--- STEP 2: DATA CLEANING ---')

print('\nChecking for missing values:')
display(df.isnull().sum().to_frame(name='Missing Values'))

# Dropping irrelevant columns (UID and Product ID)
print('\nDropping UDI and Product ID columns...')
df = df.drop(columns=['UDI', 'Product ID'])
print('Columns after dropping:', df.columns.tolist())

print('\nVerifying data types after cleaning:')
df.info()

print('--- STEP 3: Exploratory Data Analysis ---')

sns.set_style("whitegrid")


# 1. Class Balance
def plot_class_balance(df):
    print("\n===== Class Balance =====")

    failure_count = df["Machine failure"].value_counts()
    failure_pct = df["Machine failure"].value_counts(normalize=True) * 100

    print("Counts:")
    print(failure_count)

    print("\nPercentages:")
    print(failure_pct.round(2))

    plt.figure(figsize=(6, 4))

    ax = sns.countplot(
        x="Machine failure",
        hue="Machine failure",
        data=df,
        palette=["#4C72B0", "#DD8452"],
        legend=False
    )

    plt.title("Class Balance: Failure vs No Failure")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.ylabel("Count")

    total = len(df)

    for p in ax.patches:
        pct = 100 * p.get_height() / total
        ax.annotate(
            f"{pct:.1f}%",
            (p.get_x() + p.get_width()/2, p.get_height()),
            ha="center",
            va="bottom"
        )

    plt.show()


# 2. Failure Type Breakdown
def plot_failure_types(df):

    print("\n===== Failure Type Breakdown =====")

    failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    failure_type_counts = df[failure_types].sum().sort_values(ascending=False)

    print(failure_type_counts)

    plt.figure(figsize=(7,4))

    sns.barplot(
        x=failure_type_counts.index,
        y=failure_type_counts.values,
        hue=failure_type_counts.index,
        palette="mako",
        legend=False
    )

    plt.title("Failure Type Frequency")
    plt.xlabel("Failure Type")
    plt.ylabel("Count")
    plt.show()

# 3. Feature Distributions
def plot_feature_distributions(df):

    print("\n===== Feature Distributions =====")

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12,8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):

        sns.histplot(
            df[col],
            kde=True,
            bins=40,
            color="#4C72B0",
            ax=axes[i]
        )

        axes[i].set_title(f"Distribution of {col}")

    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()

# 4. Feature vs Failure
def plot_feature_vs_failure(df):

    print("\n===== Feature vs Failure =====")

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    fig, axes = plt.subplots(2,3, figsize=(12,8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):

        sns.boxplot(
            x="Machine failure",
            y=col,
            hue="Machine failure",
            data=df,
            palette=["#4C72B0","#DD8452"],
            legend=False,
            ax=axes[i]
        )

        axes[i].set_title(f"{col} by Failure Status")

    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()


# 5. Correlation Heatmap
def plot_correlation_heatmap(df):

    print("\n===== Correlation Heatmap =====")

    numeric_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ]

    corr_cols = numeric_cols + [
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]

    corr_matrix = df[corr_cols].corr()

    print(corr_matrix.round(2))

    plt.figure(figsize=(8,6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Heatmap")

    plt.show()

# 6. Scatter Relationships
def plot_scatter_relationships(df):

    print("\n===== Key Scatter Relationships =====")

    fig, axes = plt.subplots(1,3, figsize=(12,5))

    sns.scatterplot(
        x="Air temperature [K]",
        y="Process temperature [K]",
        hue="Machine failure",
        data=df,
        alpha=0.5,
        palette=["#4C72B0","#DD8452"],
        ax=axes[0]
    )

    axes[0].set_title("Air Temp vs Process Temp")

    sns.scatterplot(
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue="Machine failure",
        data=df,
        alpha=0.5,
        palette=["#4C72B0","#DD8452"],
        ax=axes[1]
    )

    axes[1].set_title("Rotational Speed vs Torque")

    sns.scatterplot(
        x="Tool wear [min]",
        y="Torque [Nm]",
        hue="Machine failure",
        data=df,
        alpha=0.5,
        palette=["#4C72B0","#DD8452"],
        ax=axes[2]
    )

    axes[2].set_title("Tool Wear vs Torque")

    plt.tight_layout()
    plt.show()

# 7. Product Type vs Failure Rate
def plot_product_type_failure(df):

    print("\n===== Product Type vs Failure Rate =====")

    type_failure_rate = (
        df.groupby("Type")["Machine failure"]
        .mean() * 100
    )

    print(type_failure_rate.round(2))

    plt.figure(figsize=(6,4))

    sns.barplot(
        x=type_failure_rate.index,
        y=type_failure_rate.values,
        hue=type_failure_rate.index,
        palette="viridis",
        legend=False
    )

    plt.title("Failure Rate (%) by Product Quality Type")
    plt.xlabel("Product Type")
    plt.ylabel("Failure Rate (%)")

    plt.show()


# Run Complete EDA
def run_eda(df):

    plot_class_balance(df)
    plot_failure_types(df)
    plot_feature_distributions(df)
    plot_feature_vs_failure(df)
    plot_correlation_heatmap(df)
    plot_scatter_relationships(df)
    plot_product_type_failure(df)

print('--- STEP 4: PREPARE FEATURES & TARGET ---')

# 1. Encode the Categorical Column ('Type')
# We'll use pandas get_dummies for one-hot encoding.
# drop_first=True prevents the "dummy variable trap" by dropping one of the categories.
# dtype=int ensures the output is 0 and 1 instead of True and False.
print("\nEncoding the 'Type' column (L, M, H)...")
df_encoded = pd.get_dummies(df, columns=['Type'], drop_first=True, dtype=int)
print("Columns after one-hot encoding:\n", df_encoded.columns.tolist())

# 2. Define the Targets (y)
print("\nSeparating targets from features...")
# Binary target for initial modeling
y_binary = df_encoded['Machine failure']

# Multiclass/Multi-label targets for later experimentation
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
y_multi = df_encoded[failure_types]

# 3. Define the Features (X)

columns_to_drop = ['Machine failure'] + failure_types
X = df_encoded.drop(columns=columns_to_drop)

# 4. Verify the Splits
print('\n--- Feature Set (X) ---')
display(X.head())
print(f"Shape of X: {X.shape}")

print('\n--- Binary Target (y_binary) ---')
display(y_binary.head().to_frame())
print(f"Shape of y_binary: {y_binary.shape}")

print('\n--- Failure Type Targets (y_multi) ---')
display(y_multi.head())
print(f"Shape of y_multi: {y_multi.shape}")
