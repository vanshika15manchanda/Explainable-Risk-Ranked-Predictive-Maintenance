"""
Exploratory Data Analysis (EDA) Script
Explainable-Risk-Ranked-Predictive-Maintenance Project

Run this from the root of the repository:
    python eda.py

Reads:  data/ai4i2020.csv
Saves plots to: results/eda_plots/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/ai4i2020.csv"
OUTPUT_DIR = "results/eda_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")

print("Loading dataset...\n")
df = pd.read_csv(DATA_PATH)


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

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_balance.png")
    plt.show()


# 2. Failure Type Breakdown
def plot_failure_types(df):

    print("\n===== Failure Type Breakdown =====")

    failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    failure_type_counts = df[failure_types].sum().sort_values(ascending=False)

    print(failure_type_counts)

    plt.figure(figsize=(7, 4))

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
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/failure_types.png")
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

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
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
    plt.savefig(f"{OUTPUT_DIR}/feature_distributions.png")
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

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):

        sns.boxplot(
            x="Machine failure",
            y=col,
            hue="Machine failure",
            data=df,
            palette=["#4C72B0", "#DD8452"],
            legend=False,
            ax=axes[i]
        )

        axes[i].set_title(f"{col} by Failure Status")

    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_vs_failure.png")
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

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.show()


# 6. Scatter Relationships
def plot_scatter_relationships(df):

    print("\n===== Key Scatter Relationships =====")

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    sns.scatterplot(
        x="Air temperature [K]",
        y="Process temperature [K]",
        hue="Machine failure",
        data=df,
        alpha=0.5,
        palette=["#4C72B0", "#DD8452"],
        ax=axes[0]
    )

    axes[0].set_title("Air Temp vs Process Temp")

    sns.scatterplot(
        x="Rotational speed [rpm]",
        y="Torque [Nm]",
        hue="Machine failure",
        data=df,
        alpha=0.5,
        palette=["#4C72B0", "#DD8452"],
        ax=axes[1]
    )

    axes[1].set_title("Rotational Speed vs Torque")

    sns.scatterplot(
        x="Tool wear [min]",
        y="Torque [Nm]",
        hue="Machine failure",
        data=df,
        alpha=0.5,
        palette=["#4C72B0", "#DD8452"],
        ax=axes[2]
    )

    axes[2].set_title("Tool Wear vs Torque")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_relationships.png")
    plt.show()


# 7. Product Type vs Failure Rate
def plot_product_type_failure(df):

    print("\n===== Product Type vs Failure Rate =====")

    type_failure_rate = (
        df.groupby("Type")["Machine failure"]
        .mean() * 100
    )

    print(type_failure_rate.round(2))

    plt.figure(figsize=(6, 4))

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
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/product_type_failure.png")
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

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_eda(df)
