import matplotlib
matplotlib.use('Agg')  # Force backend that works in headless environments

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def load_data(filename):
    """Load and clean the dataset."""
    df = pd.read_csv(filename)
    df = df.dropna(axis=1, how='all')  # Drop empty columns
    df = df.dropna(subset=["Hogwarts House"])  # Keep only labeled data
    return df

def pearson_correlation_pairwise(x, y):
    """Compute Pearson correlation using pairwise non-NaN matching."""
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

def find_best_features_by_correlation(df, top_k=4):
    """Automatically find top-k most correlated features to any Hogwarts House."""
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    features = numeric_df.columns
    target = df["Hogwarts House"]
    scores = {}
    for feature in features:
        house_values = {}
        for house in target.unique():
            binary_target = [1 if h == house else 0 for h in target]
            corr = abs(pearson_correlation_pairwise(numeric_df[feature], binary_target))
            house_values[house] = corr
        scores[feature] = max(house_values.values())
    best_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [name for name, _ in best_features]

def plot_auto_pairplot(df, features, corner=True):
    """Generate and save the pair plot using top features."""
    selected = df[features + ["Hogwarts House"]].dropna()
    pair = sns.pairplot(
        selected,
        hue="Hogwarts House",
        palette="Set1",
        corner=corner,
        plot_kws={"alpha": 0.6, "s": 20}
    )
    pair.fig.suptitle("Pair Plot of Most Relevant Features", y=1.02)
    pair.savefig("pair_plot.png")
    print("✅ Plot saved to pair_plot.png")

def plot_manual_pairplot(df):
    """Plot pairplot using all numeric features manually (for full scatter matrix)."""
    df_numeric = df.select_dtypes(include=["float64", "int64"]).copy()
    df_numeric["Hogwarts House"] = df["Hogwarts House"]
    pair = sns.pairplot(
        df_numeric,
        hue="Hogwarts House",
        palette="Set1",
        corner=False,
        plot_kws={"alpha": 0.5, "s": 15}
    )
    pair.fig.suptitle("Full Pair Plot of All Features by Hogwarts House", y=1.02)
    pair.savefig("pair_plot_manual.png")
    print("✅ Manual pair plot saved to pair_plot_manual.png")

def main(mode='auto'):
    print("STARTED")
    if len(sys.argv) < 2:
        print("Usage: python pair_plot.py dataset_train.csv [--manual|--full]")
        sys.exit(1)

    filename = sys.argv[1]
    df = load_data(filename)

    if len(sys.argv) == 3 and sys.argv[2] == '--manual':
        plot_manual_pairplot(df)
    else:
        top_features = find_best_features_by_correlation(df)
        print(f"Selected top features: {top_features}")
        plot_auto_pairplot(df, top_features, corner=True)

if __name__ == "__main__":
    main()