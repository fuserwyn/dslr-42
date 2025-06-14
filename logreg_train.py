import pandas as pd
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt

# ========== Hyperparameters ==========
LEARNING_RATE = 0.001
NUM_ITERATIONS = 10000
EPSILON = 1e-8
BATCH_SIZE = 32

# ========== Utility Functions ==========

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    h = np.clip(h, EPSILON, 1 - EPSILON)
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def batch_gradient_descent(X, y, theta, lr, iterations, cost_log):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= lr * gradient
        if i % 1000 == 0:
            c = cost_function(X, y, theta)
            cost_log.append((i, c))
            print(f"[BATCH] Iteration {i} - Cost: {c:.4f}")
    return theta

def stochastic_gradient_descent(X, y, theta, lr, iterations, cost_log):
    m = len(y)
    for i in range(iterations):
        for j in range(m):
            xi = X[j].reshape(1, -1)
            yi = y[j]
            hi = sigmoid(np.dot(xi, theta))
            hi = np.clip(hi, EPSILON, 1 - EPSILON)
            gradient = np.dot(xi.T, (hi - yi))
            theta -= lr * gradient.flatten()
        if i % 100 == 0:
            c = cost_function(X, y, theta)
            cost_log.append((i, c))
            print(f"[SGD] Iteration {i} - Cost: {c:.4f}")
    return theta

def mini_batch_gradient_descent(X, y, theta, lr, iterations, batch_size, cost_log):
    m = len(y)
    for i in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for start in range(0, m, batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]
            hb = sigmoid(np.dot(xb, theta))
            hb = np.clip(hb, EPSILON, 1 - EPSILON)
            gradient = (1 / len(yb)) * np.dot(xb.T, (hb - yb))
            theta -= lr * gradient
        if i % 1000 == 0:
            c = cost_function(X, y, theta)
            cost_log.append((i, c))
            print(f"[MINIBATCH] Iteration {i} - Cost: {c:.4f}")
    return theta

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = EPSILON
    return (X - mean) / std, mean, std

def one_vs_all(X, y, num_classes, mode="batch"):
    m, n = X.shape
    all_theta = np.zeros((num_classes, n))
    cost_logs = []
    for i in range(num_classes):
        print(f"\nTraining classifier for class {i} using {mode.upper()} GD...")
        binary_y = (y == i).astype(int)
        theta = np.zeros(n)
        cost_log = []
        if mode == "sgd":
            theta = stochastic_gradient_descent(X, binary_y, theta, LEARNING_RATE, NUM_ITERATIONS, cost_log)
        elif mode == "minibatch":
            theta = mini_batch_gradient_descent(X, binary_y, theta, LEARNING_RATE, NUM_ITERATIONS, BATCH_SIZE, cost_log)
        else:
            theta = batch_gradient_descent(X, binary_y, theta, LEARNING_RATE, NUM_ITERATIONS, cost_log)
        all_theta[i] = theta
        cost_logs.append((f"Class {i}", cost_log))
    return all_theta, cost_logs

def plot_costs(cost_logs, mode):
    for label, log in cost_logs:
        iters, costs = zip(*log)
        plt.plot(iters, costs, label=label)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title(f"Training Cost vs Iteration ({mode.upper()} GD)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"cost_plot_{mode}.png")
    print(f"📈 Cost plot saved as cost_plot_{mode}.png")
    plt.clf()

# ========== Main Training Logic ==========

def main():
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py dataset_train.csv [--mode sgd|batch|minibatch]")
        sys.exit(1)

    filename = sys.argv[1]
    mode = "batch"

    if "--mode" in sys.argv:
        try:
            mode_index = sys.argv.index("--mode") + 1
            mode = sys.argv[mode_index].lower()
            if mode not in ["sgd", "batch", "minibatch"]:
                print("Invalid mode. Use 'batch', 'sgd', or 'minibatch'.")
                sys.exit(1)
        except IndexError:
            print("Missing value for --mode.")
            sys.exit(1)

    df = pd.read_csv(filename)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=["Hogwarts House"])

    selected_features = [
        "Herbology",
        "Charms",
        "Ancient Runes",
        "Potions",
        "Defense Against the Dark Arts"
    ]

    X = df[selected_features].copy()
    X = X.fillna(X.mean(numeric_only=True))
    X = X.to_numpy()

    y_raw = df["Hogwarts House"].to_numpy()
    house_to_int = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
    y = np.array([house_to_int[house] for house in y_raw])

    X, mean, std = normalize_features(X)
    X = np.clip(X, -10, 10)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print("🔎 Sanity check: X stats")
    print("  Min:", np.min(X))
    print("  Max:", np.max(X))
    print("  Mean:", np.mean(X))
    print("  Std:", np.std(X))
    print("  Any NaN?", np.isnan(X).any())
    print("  Any Inf?", np.isinf(X).any())

    X = np.c_[np.ones(X.shape[0]), X]

    print(f"Starting training using {mode.upper()} gradient descent...")
    all_theta, cost_logs = one_vs_all(X, y, num_classes=4, mode=mode)

    with open("weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Bias"] + selected_features
        writer.writerow(["House"] + header)
        for house, theta in zip(house_to_int.keys(), all_theta):
            writer.writerow([house] + list(theta))

    print("✅ Training complete. Weights saved to weights.csv.")
    plot_costs(cost_logs, mode)

if __name__ == "__main__":
    main()
