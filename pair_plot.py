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

def plot_pairplot(df):
    """Generate and save the pair plot."""
    # Select numeric columns and add the target label
    numeric_df = df.select_dtypes(include=["float64", "int64"]).copy()
    numeric_df["Hogwarts House"] = df["Hogwarts House"]

    # Generate the pair plot
    pair = sns.pairplot(
        numeric_df,
        hue="Hogwarts House",
        palette="Set1",
        corner=True,
        plot_kws={"alpha": 0.6, "s": 20}
    )

    # Add title and save the figure
    pair.fig.suptitle("Pair Plot of Features by Hogwarts House", y=1.02)
    output_path = "pair_plot.png"
    try:
        pair.savefig("pair_plot.png")
        print("✅ Plot saved to pair_plot.png")
    except Exception as e:
        print(f"❌ Failed to save plot: {e}")


def main():
    print("STARTED")
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py dataset_train.csv")
        sys.exit(1)


    filename = sys.argv[1]
    df = load_data(filename)
    plot_pairplot(df)

if __name__ == "__main__":
    main()
