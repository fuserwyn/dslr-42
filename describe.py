import csv
import sys
import math
from collections import Counter


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_csv(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
    headers = data[0]
    rows = data[1:]
    return headers, rows

def extract_numeric_columns(headers, rows):
    total_rows = len(rows)
    columns = {header: [] for header in headers}
    numeric_counts = {header: 0 for header in headers}

    for row in rows:
        for i, value in enumerate(row):
            if is_float(value):
                columns[headers[i]].append(float(value))
                numeric_counts[headers[i]] += 1

    numeric_columns = {
        key: val for key, val in columns.items()
        if numeric_counts[key] / total_rows >= 0.9
    }
    return numeric_columns

def std(data, m):
    # Standard deviation:
    # σ = sqrt( (1/n) * Σ (xᵢ - μ)² )
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))

def percentile(data, p):
    # Percentile: shows the value below which a given percentage of observations fall.
    # For example, the 25th percentile is the value below which 25% of the data lies.
    # It helps understand the spread and distribution of the dataset.
    # Calculation uses linear interpolation:
    # 1. Sort the data
    # 2. Compute index k = (n - 1) * (p / 100)
    # 3. Interpolate between data[k_floor] and data[k_ceil]
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c - k)
    d1 = data_sorted[int(c)] * (k - f)
    return d0 + d1

def mode(data):
    # Mode: the value that appears most frequently
    count = Counter(data)
    return count.most_common(1)[0][0]

def skewness(data, m, s):
    # Skewness:
    # skew = (1/n) * Σ( (xᵢ - μ)^3 ) / σ^3
    # Measures asymmetry of the distribution
    n = len(data)
    return sum((x - m)**3 for x in data) / (n * (s**3)) if s != 0 else 0

def kurtosis(data, m, s):
    # Excess Kurtosis:
    # kurtosis = (1/n) * Σ( (xᵢ - μ)^4 ) / σ^4 - 3
    # Measures the tailedness of the distribution
    n = len(data)
    return sum((x - m)**4 for x in data) / (n * (s**4)) - 3 if s != 0 else -3

def describe(columns):
    print(f"{'Feature':<30} {'Count':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'25%':>12} {'50%':>12} {'75%':>12} {'Max':>12}")
    for feature, values in columns.items():
        if len(values) == 0:
            continue
        cnt = len(values)
        mn = sum(values) / cnt
        sd = std(values, mn)
        mi = min(values)
        ma = max(values)
        p25 = percentile(values, 25)
        p50 = percentile(values, 50)
        p75 = percentile(values, 75)
        print(f"{feature:<30} {cnt:8.0f} {mn:12.3f} {sd:12.6f} {mi:12.6f} {p25:12.6f} {p50:12.6f} {p75:12.6f} {ma:12.3f}")

def describe_bonus(columns):
    print("\n🎁 Bonus Statistics:")
    for feature, values in columns.items():
        if len(values) == 0:
            continue
        cnt = len(values)
        mn = sum(values) / cnt
        sd = std(values, mn)
        rng = max(values) - min(values)
        mod = mode(values)
        sk = skewness(values, mn, sd)
        kurt = kurtosis(values, mn, sd)
        print(f"\nFeature: {feature}")
        print(f"  Range:            {rng:.4f}")
        print(f"  Mode:             {mod}")
        print(f"  Skewness:         {sk:.4f}")
        print(f"  Kurtosis:         {kurt:.4f}")

def main():
    print("Script started!")
    if len(sys.argv) < 2:
        print("Usage: python describe.py dataset_train.csv [--bonus]")
        sys.exit(1)

    filename = sys.argv[1]
    enable_bonus = False
    if len(sys.argv) == 3 and sys.argv[2] == "--bonus":
        enable_bonus = True

    headers, rows = parse_csv(filename)
    numeric_columns = extract_numeric_columns(headers, rows)
    describe(numeric_columns)
    if enable_bonus:
        describe_bonus(numeric_columns)

if __name__ == "__main__":
    main()
