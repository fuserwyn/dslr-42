import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def load_data(filename):
    """Load the CSV and return a cleaned DataFrame."""
    df = pd.read_csv(filename)
    # Remove columns that are fully empty
    df = df.dropna(axis=1, how='all')
    # Drop rows with no house assigned
    df = df.dropna(subset=["Hogwarts House"])
    return df

def plot_histograms(df, save=False):
    """
    For each numerical feature, plot a histogram of values per house.
    This helps identify which courses show similar distributions across houses.
    If `save` is True, saves each histogram as a PNG file instead of showing.
    """
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
    houses = df["Hogwarts House"].unique()
    house_colors = {
        "Gryffindor": "red",
        "Hufflepuff": "gold",
        "Ravenclaw": "blue",
        "Slytherin": "green"
    }

    output_dir = "histograms"
    if save:
        os.makedirs(output_dir, exist_ok=True)

    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        for house in houses:
            house_data = df[df["Hogwarts House"] == house][feature].dropna()
            plt.hist(house_data, bins=20, alpha=0.5, label=house, color=house_colors.get(house, None))

        plt.title(f"Distribution of {feature} by House")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save:
            filename = os.path.join(output_dir, f"histogram_{feature}.png")
            plt.savefig(filename)
            print(f"✅ Saved: {filename}")
            plt.close()
        else:
            plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python histogram.py dataset_train.csv [--save]")
        sys.exit(1)

    filename = sys.argv[1]
    save_flag = "--save" in sys.argv

    df = load_data(filename)
    plot_histograms(df, save=save_flag)

if __name__ == "__main__":
    main()
