import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def load_data(filename):
    """Load and clean the dataset."""
    df = pd.read_csv(filename)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=["Hogwarts House"])
    return df

def find_most_similar_features(df):
    """Find the two most correlated numeric features using Pearson correlation."""
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    correlation_matrix = numeric_df.corr().abs()

    # Create a boolean mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    upper_triangle = correlation_matrix.where(mask)

    # Find the pair with the highest correlation
    max_corr = upper_triangle.max().max()
    most_similar_pair = upper_triangle.stack().idxmax()
    return most_similar_pair, max_corr

def plot_scatter(df, feature1, feature2):
    """Save the scatter plot as a PNG instead of displaying it."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature1, y=feature2, hue="Hogwarts House", palette="Set1", alpha=0.7)
    plt.title(f"Scatter Plot: {feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot to a file instead of showing it
    filename = f"scatter_{feature1}_vs_{feature2}.png".replace(" ", "_")
    plt.savefig(filename)
    print(f"✅ Scatter plot saved as {filename}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = load_data(filename)
    (f1, f2), corr = find_most_similar_features(df)
    print(f"Most similar features: {f1} vs {f2} (correlation = {corr:.4f})")
    plot_scatter(df, f1, f2)

if __name__ == "__main__":
    main()
