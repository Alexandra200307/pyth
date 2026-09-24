import json, csv


def matmul(A, B):
    n, p = len(B), len(B[0])
    C = [[0.0] * p for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matvec(A, v):
    n = len(A[0])
    return [sum(A[i][j] * v[j] for j in range(n)) for i in range(len(A))]


def trace(A):
    n = len(A)
    return sum(A[i][i] for i in range(n))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def histogram(data, bins=10):
    if not data:
        return [0] * bins
    lo, hi = min(data), max(data)
    if hi == lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for x in data:
        idx = int((x - lo) / width)
        if idx == bins:
            idx -= 1
        if 0 <= idx < bins:
            counts[idx] += 1
    return counts


def kernel_filter(data, kernel):
    n, k = len(data), len(kernel)
    r = k // 2
    out = [0.0] * n
    for i in range(n):
        for j in range(k):
            idx = i + j - r
            if 0 <= idx < n:
                out[i] += data[idx] * kernel[j]
    return out


def read_data(filename):
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "json":
        with open(filename, encoding="utf-8") as f:
            d = json.load(f)
        if d and isinstance(d[0], list):
            return [[float(x) for x in r] for r in d]
        return [float(x) for x in d]

    if ext == "csv":
        with open(filename, encoding="utf-8", newline="") as f:
            rows = [r for r in csv.reader(f) if r]
        if len(rows) == 1:
            return [float(x) for x in rows[0]]
        if all(len(r) == 1 for r in rows):
            return [float(r[0]) for r in rows]
        return [[float(x) for x in r] for r in rows]

    with open(filename, encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]
    if all(len(r) == 1 for r in rows):
        return [float(r[0]) for r in rows]
    return [[float(x) for x in r] for r in rows]


def write_data(filename, data):
    ext = filename.rsplit(".", 1)[-1].lower()
    is_mat = bool(data) and isinstance(data[0], (list, tuple))

    if ext == "json":
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)

    elif ext == "csv":
        with open(filename, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerows(data if is_mat else [[x] for x in data])

    else:
        with open(filename, "w", encoding="utf-8") as f:
            if is_mat:
                for r in data:
                    f.write(" ".join(map(str, r)) + "\n")
            else:
                for x in data:
                    f.write(f"{x}\n")