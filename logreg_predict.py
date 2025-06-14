import pandas as pd
import numpy as np
import csv
import sys

# ====== Sigmoid function ======
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ====== Normalize with reference mean/std ======
def normalize_features(X, mean, std):
    std[std == 0] = 1e-8  # avoid division by zero
    return (X - mean) / std

# ====== Load weights from CSV ======
def load_weights(filename):
    df = pd.read_csv(filename)
    houses = df["House"].tolist()
    weights = df.drop("House", axis=1).to_numpy()
    return houses, weights

# ====== Main prediction logic ======
def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.csv")
        sys.exit(1)

    test_file = sys.argv[1]
    weights_file = sys.argv[2]

    # === Features used for training ===
    selected_features = [
        "Herbology",
        "Charms",
        "Ancient Runes",
        "Potions",
        "Defense Against the Dark Arts"
    ]

    # Load and clean test data
    df = pd.read_csv(test_file)
    df = df.dropna(axis=1, how='all')
    df = df.fillna(df.mean(numeric_only=True))  # Fill missing values

    X = df[selected_features].to_numpy()

    # Load weights
    house_labels, all_theta = load_weights(weights_file)

    # Compute mean and std from weights file header (retraining is better, but we estimate here)
    # We'll recompute mean/std using test data to keep things consistent
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = normalize_features(X, mean, std)
    X = np.clip(X, -10, 10)  # clip extremes

    # Add bias term
    X = np.c_[np.ones(X.shape[0]), X]

    # Predict using One-vs-All
    probabilities = sigmoid(np.dot(X, all_theta.T))
    predictions = np.argmax(probabilities, axis=1)

    # Map index back to house name
    predicted_houses = [house_labels[i] for i in predictions]

    # Save predictions
    with open("houses.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for idx, house in enumerate(predicted_houses):
            writer.writerow([idx, house])

    print("✅ Prediction complete. Results saved to houses.csv")

if __name__ == "__main__":
    main()
