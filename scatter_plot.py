import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os


def load_data(filename):
    """Load and clean the dataset."""
    df = pd.read_csv(filename)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=["Hogwarts House"])
    return df


def pearson_correlation_pairwise(x, y):
    """
    Compute Pearson correlation using pairwise non-NaN matching like pandas.corr().
    """
    paired = [(xi, yi) for xi, yi in zip(x, y) if not pd.isna(xi) and not pd.isna(yi)]
    if not paired:
        return float('nan')
    x_clean, y_clean = zip(*paired)
    n = len(x_clean)
    mean_x = sum(x_clean) / n
    mean_y = sum(y_clean) / n
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x_clean, y_clean))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x_clean)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y_clean)
    denominator = (denominator_x * denominator_y) ** 0.5
    if denominator == 0:
        return 0
    return numerator / denominator


def find_top_k_similar_features(df, k=5):
    """Find the top-k most correlated numeric feature pairs using manual Pearson correlation."""
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    features = numeric_df.columns
    correlations = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            x, y = numeric_df[f1], numeric_df[f2]
            corr = abs(pearson_correlation_pairwise(x, y))
            correlations.append(((f1, f2), corr))

    top_k = sorted(correlations, key=lambda x: x[1], reverse=True)[:k]
    return top_k


def plot_scatter(df, feature1, feature2, output_dir):
    """Save the scatter plot as a PNG in a specified directory."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=feature1, y=feature2, hue="Hogwarts House", palette="Set1", alpha=0.7)
    plt.title(f"Scatter Plot: {feature1} vs {feature2}")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    plt.tight_layout()

    filename = f"scatter_{feature1}_vs_{feature2}.png".replace(" ", "_")
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"✅ Scatter plot saved as {filepath}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = load_data(filename)

    output_dir = "best_scatter_pairs"
    os.makedirs(output_dir, exist_ok=True)

    top_pairs = find_top_k_similar_features(df, k=5)
    for (f1, f2), corr in top_pairs:
        print(f"Pair: {f1} vs {f2} (correlation = {corr:.4f})")
        plot_scatter(df, f1, f2, output_dir)


if __name__ == "__main__":
    main()
    