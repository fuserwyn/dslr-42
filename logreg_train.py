import pandas as pd
import sys
import csv
import math
import random
import matplotlib.pyplot as plt

# ========== Hyperparameters ==========
LEARNING_RATE = 0.001
NUM_ITERATIONS = 10000
EPSILON = 1e-8
BATCH_SIZE = 32

# ========== Utility Functions ==========

def sigmoid(z):
    return [1 / (1 + math.exp(-x)) for x in z]

def dot(X, theta):
    return [sum(xi * ti for xi, ti in zip(x, theta)) for x in X]

def cost_function(X, y, theta):
    h = sigmoid(dot(X, theta))
    cost = 0.0
    m = len(y)
    for hi, yi in zip(h, y):
        hi = min(max(hi, EPSILON), 1 - EPSILON)
        cost += -(yi * math.log(hi) + (1 - yi) * math.log(1 - hi))
    return cost / m

def batch_gradient_descent(X, y, theta, lr, iterations, cost_log):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(dot(X, theta))
        gradient = [0] * len(theta)
        for j in range(len(theta)):
            gradient[j] = sum((h[k] - y[k]) * X[k][j] for k in range(m)) / m
        theta = [theta[j] - lr * gradient[j] for j in range(len(theta))]
        if i % 1000 == 0:
            c = cost_function(X, y, theta)
            cost_log.append((i, c))
            print(f"[BATCH] Iteration {i} - Cost: {c:.4f}")
    return theta

def stochastic_gradient_descent(X, y, theta, lr, iterations, cost_log):
    m = len(y)
    for i in range(iterations):
        for j in range(m):
            xi = X[j]
            yi = y[j]
            hi = 1 / (1 + math.exp(-sum(x * t for x, t in zip(xi, theta))))
            hi = min(max(hi, EPSILON), 1 - EPSILON)
            gradient = [(hi - yi) * x for x in xi]
            theta = [t - lr * g for t, g in zip(theta, gradient)]
        if i % 100 == 0:
            c = cost_function(X, y, theta)
            cost_log.append((i, c))
            print(f"[SGD] Iteration {i} - Cost: {c:.4f}")
    return theta

def mini_batch_gradient_descent(X, y, theta, lr, iterations, batch_size, cost_log):
    m = len(y)
    for i in range(iterations):
        indices = list(range(m))
        random.shuffle(indices)
        for start in range(0, m, batch_size):
            end = start + batch_size
            batch = indices[start:end]
            xb = [X[k] for k in batch]
            yb = [y[k] for k in batch]
            hb = sigmoid(dot(xb, theta))
            hb = [min(max(h, EPSILON), 1 - EPSILON) for h in hb]
            gradient = [0] * len(theta)
            for j in range(len(theta)):
                gradient[j] = sum((hb[k] - yb[k]) * xb[k][j] for k in range(len(yb))) / len(yb)
            theta = [theta[j] - lr * gradient[j] for j in range(len(theta))]
        if i % 1000 == 0:
            c = cost_function(X, y, theta)
            cost_log.append((i, c))
            print(f"[MINIBATCH] Iteration {i} - Cost: {c:.4f}")
    return theta

def normalize_features(X):
    cols = list(zip(*X))
    mean = [sum(col) / len(col) for col in cols]
    std = [
        math.sqrt(sum((x - m) ** 2 for x in col) / len(col)) if sum((x - m) ** 2 for x in col) > 0 else EPSILON
        for col, m in zip(cols, mean)
    ]
    X_norm = [[(x - m) / s for x, m, s in zip(row, mean, std)] for row in X]
    return X_norm, mean, std

def one_vs_all(X, y, num_classes, mode="batch"):
    n = len(X[0])
    all_theta = []
    cost_logs = []
    for i in range(num_classes):
        print(f"Training classifier for class {i} using {mode.upper()} GD...")
        binary_y = [1 if val == i else 0 for val in y]
        theta = [0.0] * n
        cost_log = []
        if mode == "sgd":
            theta = stochastic_gradient_descent(X, binary_y, theta, LEARNING_RATE, NUM_ITERATIONS, cost_log)
        elif mode == "minibatch":
            theta = mini_batch_gradient_descent(X, binary_y, theta, LEARNING_RATE, NUM_ITERATIONS, BATCH_SIZE, cost_log)
        else:
            theta = batch_gradient_descent(X, binary_y, theta, LEARNING_RATE, NUM_ITERATIONS, cost_log)
        all_theta.append(theta)
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

# ========== Main ==========

def main():
    if len(sys.argv) < 2:
        print("Usage: python logreg_train.py dataset_train.csv [--mode sgd|batch|minibatch]")
        sys.exit(1)

    filename = sys.argv[1]
    mode = "batch"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode") + 1
        if idx < len(sys.argv):
            mode = sys.argv[idx].lower()

    df = pd.read_csv(filename)
    df = df.dropna(axis=1, how='all')
    df = df.dropna(subset=["Hogwarts House"])

    features = ["Herbology", "Charms", "Ancient Runes", "Potions", "Defense Against the Dark Arts"]
    df = df.fillna(df.mean(numeric_only=True))
    X = df[features].values.tolist()
    y_raw = df["Hogwarts House"].tolist()
    y_map = {"Gryffindor": 0, "Hufflepuff": 1, "Ravenclaw": 2, "Slytherin": 3}
    y = [y_map[house] for house in y_raw]

    X, mean, std = normalize_features(X)
    X = [[max(min(x, 10), -10) for x in row] for row in X]
    X = [[1.0] + row for row in X]

    print("🔎 Sanity check:")
    flat = sum(X, [])
    print("  Min:", min(flat))
    print("  Max:", max(flat))
    print("  Mean:", sum(flat) / len(flat))
    print("  Any NaN?", any(math.isnan(x) for x in flat))
    print("  Any Inf?", any(math.isinf(x) for x in flat))

    all_theta, cost_logs = one_vs_all(X, y, 4, mode)

    with open("weights.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["House", "Bias"] + features)
        for house, theta in zip(y_map.keys(), all_theta):
            writer.writerow([house] + theta)

    print("✅ Training complete. Weights saved to weights.csv.")
    plot_costs(cost_logs, mode)

if __name__ == "__main__":
    main()
