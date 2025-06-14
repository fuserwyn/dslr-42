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

def mean(data):
    return sum(data) / len(data)

def std(data, m):
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))

def percentile(data, p):
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
    count = Counter(data)
    return count.most_common(1)[0][0]

def skewness(data, m, s):
    n = len(data)
    return sum((x - m)**3 for x in data) / (n * (s**3)) if s != 0 else 0

def kurtosis(data, m, s):
    n = len(data)
    return sum((x - m)**4 for x in data) / (n * (s**4)) - 3 if s != 0 else -3

def describe(columns):
    print(f"{'Feature':<20} {'Count':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10}")
    for feature, values in columns.items():
        if len(values) == 0:
            continue
        cnt = len(values)
        mn = mean(values)
        sd = std(values, mn)
        mi = min(values)
        ma = max(values)
        p25 = percentile(values, 25)
        p50 = percentile(values, 50)
        p75 = percentile(values, 75)
        print(f"{feature:<20} {cnt:10.6f} {mn:10.6f} {sd:10.6f} {mi:10.6f} {p25:10.6f} {p50:10.6f} {p75:10.6f} {ma:10.6f}")

def describe_bonus(columns):
    print("\n🎁 Bonus Statistics:")
    for feature, values in columns.items():
        if len(values) == 0:
            continue
        cnt = len(values)
        mn = mean(values)
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