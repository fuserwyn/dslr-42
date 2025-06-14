import pandas as pd
import csv
import sys
import math

# ====== Sigmoid function ======
def sigmoid(z):
    return [1 / (1 + math.exp(-val)) for val in z]

# ====== Normalize with manual mean/std ======
def manual_normalize(X):
    mean = []
    std = []
    norm_X = []
    for col in zip(*X):
        col_list = list(col)
        m = sum(col_list) / len(col_list)
        s = math.sqrt(sum((x - m) ** 2 for x in col_list) / len(col_list))
        s = s if s != 0 else 1e-8
        mean.append(m)
        std.append(s)
    for row in X:
        norm_X.append([(val - m) / s for val, m, s in zip(row, mean, std)])
    return norm_X

# ====== Load weights from CSV ======
def load_weights(filename):
    df = pd.read_csv(filename)
    houses = df["House"].tolist()
    weights = df.drop("House", axis=1).values.tolist()
    return houses, weights

# ====== Matrix multiply ======
def matmul(A, B):
    result = []
    for row in A:
        row_result = []
        for col in zip(*B):
            row_result.append(sum(r * c for r, c in zip(row, col)))
        result.append(row_result)
    return result

# ====== Main prediction logic ======
def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py dataset_test.csv weights.csv")
        sys.exit(1)

    test_file = sys.argv[1]
    weights_file = sys.argv[2]

    selected_features = [
        "Herbology",
        "Charms",
        "Ancient Runes",
        "Potions",
        "Defense Against the Dark Arts"
    ]

    df = pd.read_csv(test_file)
    df = df.dropna(axis=1, how='all')
    df = df.fillna(df.mean(numeric_only=True))

    X = df[selected_features].values.tolist()

    house_labels, all_theta = load_weights(weights_file)

    X = manual_normalize(X)
    X = [[max(min(x, 10), -10) for x in row] for row in X]

    # Add bias
    X = [[1.0] + row for row in X]

    logits = matmul(X, [list(t) for t in zip(*all_theta)])
    probs = [sigmoid(row) for row in logits]
    predictions = [row.index(max(row)) for row in probs]

    predicted_houses = [house_labels[i] for i in predictions]

    with open("houses.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Hogwarts House"])
        for idx, house in enumerate(predicted_houses):
            writer.writerow([idx, house])

    print("✅ Prediction complete. Results saved to houses.csv")

if __name__ == "__main__":
    main()