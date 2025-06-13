import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_data(filename):
    """Load the CSV and return a cleaned DataFrame."""
    df = pd.read_csv(filename)
    # Remove columns that are fully empty
    df = df.dropna(axis=1, how='all')
    # Drop rows with no house assigned
    df = df.dropna(subset=["Hogwarts House"])
    return df

def plot_histograms(df):
    """
    For each numerical feature, plot a histogram of values per house.
    This helps identify which courses show similar distributions across houses.
    """
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns
    houses = df["Hogwarts House"].unique()

    for feature in numeric_features:
        plt.figure(figsize=(10, 6))
        for house in houses:
            house_data = df[df["Hogwarts House"] == house][feature].dropna()
            plt.hist(house_data, bins=20, alpha=0.5, label=house)
        
        plt.title(f"Distribution of {feature} by House")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python histogram.py dataset_train.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = load_data(filename)
    plot_histograms(df)

if __name__ == "__main__":
    main()
