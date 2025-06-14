import pandas as pd
import numpy as np
import sys
import csv

# ========== Hyperparameters ==========
LEARNING_RATE = 0.001
NUM_ITERATIONS = 10000
EPSILON = 1e-8

# ========== Utility Functions ==========

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    """Compute logistic regression cost."""
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    h = np.clip(h, EPSILON, 1 - EPSILON)  # Prevent log(0)
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, lr, iterations):
    """Perform batch gradient descent."""
    m = len(y)
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= lr * gradient
        if i % 1000 == 0:
            print(f"Iteration {i} - Cost: {cost_function(X, y, theta):.4f}")
    return theta

def normalize_features(X):
    """Normalize features using mean and std, safely."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = EPSILON  # Prevent division by zero
    X_norm = (X - mean) / std
    return X_norm, mean, std

def one_vs_all(X, y, num_classes):
    """Train one-vs-all logistic regressions."""
    m, n = X.shape
    all_theta = np.zeros((num_classes, n))
    for i in range(num_classes):
        print(f"\nTraining classifier for class {i}...")
        binary_y = (y == i).astype(int)
        theta = np.zeros(n)
        theta = gradient_descent(X, binary_y, theta, LEARNING_RATE, NUM_ITERATIONS)
        all_theta[i] = theta
    return all_theta

# ========== Main Training Logic ==========

def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]
    df = pd.read_csv(filename)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=["Hogwarts House"])

    # === Feature Selection (You can change these later) ===
    selected_features = [
        "Herbology",
        "Charms",
        "Ancient Runes",
        "Potions",
        "Defense Against the Dark Arts"
    ]

    X = df[selected_features].to_numpy()
    y_raw = df["Hogwarts House"].to_numpy()

    # Encode labels
    house_to_int = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
    y = np.array([house_to_int[house] for house in y_raw])

    # Normalize features
    X, mean, std = normalize_features(X)

    # Clip to prevent extreme values
    X = np.clip(X, -10, 10)

    print("🔎 Checking input data...")

    if np.isnan(X).any():
        print("❌ X contains NaNs!")
        X = np.nan_to_num(X, nan=0.0)

    if np.isinf(X).any():
        print("❌ X contains inf/-inf!")
        X = np.where(np.isinf(X), 0.0, X)

    print("✅ Cleaned input data (no NaNs or infs)")


    # Add bias term (column of ones)
    X = np.c_[np.ones(X.shape[0]), X]

    # Train One-vs-All logistic regression
    print("Starting training...")
    all_theta = one_vs_all(X, y, num_classes=4)

    # Save weights
    with open("weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Bias"] + selected_features
        writer.writerow(["House"] + header)
        for house, theta in zip(house_to_int.keys(), all_theta):
            writer.writerow([house] + list(theta))

    print("✅ Training complete. Weights saved to weights.csv.")

if __name__ == "__main__":
    main()
