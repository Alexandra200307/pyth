import json
import csv
import os

import cv2
import numpy as np


def matmul(A, B):
    C = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matvec(A, v):
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def trace(A):
    return sum(A[i][i] for i in range(len(A)))


def dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def histogram(data, bins=10):
    if not data:
        return [0] * bins

    lo, hi = min(data), max(data)
    if hi == lo:
        hi = lo + 1

    step = (hi - lo) / bins
    counts = [0] * bins
    for x in data:
        idx = int((x - lo) / step)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return counts


def kernel_filter(data, kernel):
    n = len(data)
    half = len(kernel) // 2
    out = []
    for i in range(n):
        s = 0.0
        for j in range(len(kernel)):
            k = i + j - half
            if 0 <= k < n:
                s += data[k] * kernel[j]
        out.append(s)
    return out


def read_json(filename):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], list):
        return [[float(x) for x in row] for row in data]
    return [float(x) for x in data]


def read_csv(filename):
    with open(filename, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f) if r]

    if len(rows) == 1:
        return [float(x) for x in rows[0]]
    if all(len(r) == 1 for r in rows):
        return [float(r[0]) for r in rows]
    return [[float(x) for x in r] for r in rows]


def read_txt(filename):
    with open(filename, encoding="utf-8") as f:
        rows = [line.split() for line in f if line.strip()]

    if all(len(r) == 1 for r in rows):
        return [float(r[0]) for r in rows]
    return [[float(x) for x in r] for r in rows]


READERS = {"json": read_json, "csv": read_csv, "txt": read_txt}


def read_data(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return READERS.get(ext, read_txt)(filename)


def write_data(filename, data):
    ext = filename.rsplit(".", 1)[-1].lower()
    is_mat = len(data) > 0 and isinstance(data[0], (list, tuple))

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
                for row in data:
                    f.write(" ".join(str(x) for x in row) + "\n")
            else:
                for x in data:
                    f.write(str(x) + "\n")


def load_image(path, flags=cv2.IMREAD_COLOR):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, flags)
    except Exception:
        img = None

    if img is None:
        raise FileNotFoundError("не открывается файл: " + path)
    return img


def save_image(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError("не сохраняется файл: " + path)
    buf.tofile(path)


def compute_histogram(img, bins=256):
    step = 256 / bins
    channels = [img] if len(img.shape) == 2 else [img[:, :, c] for c in range(3)]

    res = []
    for ch in channels:
        h = [0] * bins
        for v in ch.flatten():
            idx = int(v / step)
            if idx >= bins:
                idx = bins - 1
            h[idx] += 1
        res.append(h)

    return res[0] if len(img.shape) == 2 else res


def _equalize_gray(img):
    hist = [0] * 256
    for v in img.flatten():
        hist[v] += 1

    cdf, s = [0] * 256, 0
    for i in range(256):
        s += hist[i]
        cdf[i] = s

    cdf_min = next((c for c in cdf if c > 0), 0)
    total = img.size

    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        if cdf[i] > cdf_min:
            table[i] = round((cdf[i] - cdf_min) / (total - cdf_min) * 255)
    return table[img]


def equalize_histogram(img):
    if len(img.shape) == 2:
        return _equalize_gray(img)

    B = img[:, :, 0].astype(np.float32)
    G = img[:, :, 1].astype(np.float32)
    R = img[:, :, 2].astype(np.float32)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 128
    Cb = (B - Y) * 0.564 + 128

    Y = _equalize_gray(Y.astype(np.uint8)).astype(np.float32)

    R2 = Y + 1.403 * (Cr - 128)
    G2 = Y - 0.714 * (Cr - 128) - 0.344 * (Cb - 128)
    B2 = Y + 1.773 * (Cb - 128)

    out = np.zeros_like(img)
    out[:, :, 0] = np.clip(B2, 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip(G2, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip(R2, 0, 255).astype(np.uint8)
    return out


def gamma_correction(img, gamma):
    if gamma <= 0:
        raise ValueError("gamma должна быть больше нуля")

    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = round(((i / 255.0) ** (1.0 / gamma)) * 255)
    return table[img]