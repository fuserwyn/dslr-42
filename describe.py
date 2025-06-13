import csv
import sys
import math

def is_float(value):
    """Check if a value can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_csv(filename):
    """Reads a CSV and returns headers and rows as lists."""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
    headers = data[0]
    rows = data[1:]
    return headers, rows

def extract_numeric_columns(headers, rows):
    """
    Returns a dict of columns containing only numeric values.
    Non-numeric data (like names or Hogwarts houses) are ignored.
    Missing values are skipped.
    """
    columns = {header: [] for header in headers}
    for row in rows:
        for i, value in enumerate(row):
            if is_float(value):
                columns[headers[i]].append(float(value))
            else:
                # Skip missing or non-numeric values
                continue
    # Keep only columns that have numeric data
    numeric_columns = {key: val for key, val in columns.items() if len(val) > 0}
    return numeric_columns

def mean(data):
    """Compute the mean manually."""
    return sum(data) / len(data)

def std(data, m):
    """Compute the standard deviation manually."""
    return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))

def percentile(data, p):
    """Compute the p-th percentile manually (0 < p < 100)."""
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c - k)
    d1 = data_sorted[int(c)] * (k - f)
    return d0 + d1

def describe(columns):
    """Computes and prints statistics for each numeric column."""
    print(f"{'Feature':<20} {'Count':>10} {'Mean':>10} {'Std':>10} {'Min':>10} {'25%':>10} {'50%':>10} {'75%':>10} {'Max':>10}")
    for feature, values in columns.items():
        if len(values) == 0:
            continue
        values_clean = values
        cnt = len(values_clean)
        mn = mean(values_clean)
        sd = std(values_clean, mn)
        mi = min(values_clean)
        ma = max(values_clean)
        p25 = percentile(values_clean, 25)
        p50 = percentile(values_clean, 50)
        p75 = percentile(values_clean, 75)
        print(f"{feature:<20} {cnt:10.6f} {mn:10.6f} {sd:10.6f} {mi:10.6f} {p25:10.6f} {p50:10.6f} {p75:10.6f} {ma:10.6f}")

def main():
    print("Script started!")
    if len(sys.argv) != 2:
        print("Usage: python describe.py dataset_train.csv")
        sys.exit(1)

    filename = sys.argv[1]
    headers, rows = parse_csv(filename)
    numeric_columns = extract_numeric_columns(headers, rows)
    describe(numeric_columns)

if __name__ == "__main__":
    main()
