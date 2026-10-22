
import argparse
import os
import random
import sys
import time
import wave

class MatrixError(Exception):
    pass


def _check_matrix(A, name="A"):
    if not isinstance(A, list) or not A:
        raise MatrixError("%s: матрица должна быть непустым списком строк" % name)
    w = len(A[0])
    for i, row in enumerate(A):
        if not isinstance(row, list) or len(row) != w:
            raise MatrixError("%s: строка %d имеет разную длину" % (name, i))


def _check_vector(v, name="v"):
    if not isinstance(v, list) or not v:
        raise MatrixError("%s: вектор должен быть непустым списком" % name)


def mat_mat_mul(A, B):
    _check_matrix(A, "A")
    _check_matrix(B, "B")
    n, m = len(A), len(A[0])
    m2, k = len(B), len(B[0])
    if m != m2:
        raise MatrixError("A: %dx%d, B: %dx%d — размеры не согласованы"
                          % (n, m, m2, k))
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt]
            for row in A]


def mat_vec_mul(A, v):
    _check_matrix(A, "A")
    _check_vector(v)
    if len(A[0]) != len(v):
        raise MatrixError("A: %dx%d, v: %d — размеры не согласованы"
                          % (len(A), len(A[0]), len(v)))
    return [sum(a * x for a, x in zip(row, v)) for row in A]


def trace(A):
    _check_matrix(A, "A")
    if len(A) != len(A[0]):
        raise MatrixError("След определён только для квадратной матрицы")
    return sum(A[i][i] for i in range(len(A)))


def dot(u, v):
    _check_vector(u, "u")
    _check_vector(v, "v")
    if len(u) != len(v):
        raise MatrixError("u: %d, v: %d — размеры не совпадают"
                          % (len(u), len(v)))
    return sum(a * b for a, b in zip(u, v))


def histogram(values, bins=10, vmin=None, vmax=None):
    _check_vector(values, "values")
    if bins <= 0:
        raise MatrixError("bins должен быть положительным")
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)
    if vmax <= vmin:
        vmax = vmin + 1e-12
    counts = [0] * bins
    step = (vmax - vmin) / bins
    for x in values:
        idx = int((x - vmin) / step)
        if idx == bins:
            idx = bins - 1
        if 0 <= idx < bins:
            counts[idx] += 1
    edges = [vmin + i * step for i in range(bins + 1)]
    return counts, edges


def kernel_filter(signal, kernel, mode="same"):
    _check_vector(signal, "signal")
    _check_vector(kernel, "kernel")
    n, k = len(signal), len(kernel)
    if mode == "valid":
        if k > n:
            raise MatrixError("Ядро длиннее сигнала")
        return [sum(signal[i + j] * kernel[j] for j in range(k))
                for i in range(n - k + 1)]
    half = k // 2
    out = []
    for i in range(n):
        s = 0.0
        for j in range(k):
            idx = i + j - half
            if 0 <= idx < n:
                s += signal[idx] * kernel[j]
        out.append(s)
    return out


def save_matrix(path, A, fmt="%.6f"):
    _check_matrix(A, "A")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("%d %d\n" % (len(A), len(A[0])))
            for row in A:
                f.write(" ".join(fmt % x for x in row) + "\n")
    except (OSError, IOError) as e:
        raise MatrixError("Ошибка записи матрицы %r: %s" % (path, e))


def load_matrix(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline().split()
            if len(head) != 2:
                raise MatrixError("Неверный заголовок файла %r" % path)
            n, m = int(head[0]), int(head[1])
            A = []
            for i in range(n):
                line = f.readline()
                if not line:
                    raise MatrixError("Неожиданный конец файла (строка %d)" % (i + 2))
                parts = line.split()
                if len(parts) != m:
                    raise MatrixError("Строка %d: ожидалось %d чисел, получено %d"
                                      % (i + 2, m, len(parts)))
                A.append([float(x) for x in parts])
    except FileNotFoundError:
        raise MatrixError("Файл %r не найден" % path)
    except (OSError, IOError) as e:
        raise MatrixError("Ошибка чтения матрицы %r: %s" % (path, e))
    return A


def save_vector(path, v, fmt="%.6f"):
    _check_vector(v)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("%d\n" % len(v))
            f.write(" ".join(fmt % x for x in v) + "\n")
    except (OSError, IOError) as e:
        raise MatrixError("Ошибка записи вектора %r: %s" % (path, e))


def load_vector(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline().split()
            if len(head) != 1:
                raise MatrixError("Неверный заголовок файла %r" % path)
            n = int(head[0])
            parts = f.readline().split()
            if len(parts) != n:
                raise MatrixError("Ожидалось %d чисел, получено %d" % (n, len(parts)))
            v = [float(x) for x in parts]
    except FileNotFoundError:
        raise MatrixError("Файл %r не найден" % path)
    except (OSError, IOError) as e:
        raise MatrixError("Ошибка чтения вектора %r: %s" % (path, e))
    return v

def _measure(func, *args, repeat=3):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times), sum(times) / len(times)


def _random_matrix(n, m):
    return [[random.random() for _ in range(m)] for _ in range(n)]


def _random_vector(n):
    return [random.random() for _ in range(n)]


def run_matrix_section():
    sizes = [10, 30, 60, 100]
    results = []

    for n in sizes:
        A = _random_matrix(n, n)
        B = _random_matrix(n, n)
        v = _random_vector(n)

        results.append(("mat_mat_mul", n) + _measure(mat_mat_mul, A, B))
        results.append(("mat_vec_mul", n) + _measure(mat_vec_mul, A, v))
        results.append(("trace", n) + _measure(trace, A))
        results.append(("dot", n) + _measure(dot, v, v))

        big = _random_vector(max(1000, n * 100))
        results.append(("histogram", len(big)) + _measure(histogram, big, 64))
        results.append(("kernel_filter", n) + _measure(kernel_filter, v, [-1, 0, 1]))

    out = "timings.txt"
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write("%-15s %8s %12s %12s\n"
                    % ("function", "n", "best, s", "avg, s"))
            for name, n, best, avg in results:
                f.write("%-15s %8d %12.6f %12.6f\n" % (name, n, best, avg))
        print("Результаты сохранены в", os.path.abspath(out))
    except (OSError, IOError) as e:
        print("[ERROR] Не удалось сохранить тайминги:", e)

    try:
        save_matrix("A.txt", _random_matrix(3, 4))
        save_vector("v.txt", _random_vector(5))
        print("A =", load_matrix("A.txt"))
        print("v =", load_vector("v.txt"))
    except MatrixError as e:
        print("[ERROR]", e)

IMAGE_EXT_COLOR = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_EXT_GRAY  = {".pgm", ".pbm"}


def _load_image(path):
    import cv2

    if not os.path.exists(path):
        raise FileNotFoundError("Файл %r не найден" % path)

    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT_COLOR:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    elif ext in IMAGE_EXT_GRAY:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Неподдерживаемое расширение: %r" % ext)

    if img is None:
        raise ValueError("OpenCV не смог прочитать %r (повреждённый файл?)" % path)
    return img


def _save_image(path, img):
    import cv2
    if not cv2.imwrite(path, img):
        raise IOError("Не удалось сохранить изображение в %r" % path)


def equalize(img):
    import cv2
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def gamma_correct(img, gamma=1.0):
    import cv2
    import numpy as np
    if gamma <= 0:
        raise ValueError("gamma должна быть > 0")
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)],
                     dtype="uint8")
    return cv2.LUT(img, table)


def compute_histogram(img, bins=256):
    import cv2
    if img.ndim == 2:
        return cv2.calcHist([img], [0], None, [bins], [0, 256])
    hists = []
    for ch in range(3):
        hists.append(cv2.calcHist([img], [ch], None, [bins], [0, 256]))
    return hists


def run_image_section(argv):
    p = argparse.ArgumentParser(
        prog="ht3.py image",
        description="Обработка изображений: эквализация, гамма, гистограмма.")
    p.add_argument("input", help="путь к изображению")
    p.add_argument("--mode", choices=["equalize", "gamma", "histogram"],
                   default="equalize")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--bins", type=int, default=256)
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args(argv)

    try:
        img = _load_image(args.input)
    except (FileNotFoundError, ValueError) as e:
        print("[ERROR] Не удалось загрузить изображение:", e)
        return

    try:
        if args.mode == "equalize":
            result = equalize(img)
            out = args.output or "out_equalized.png"
            _save_image(out, result)
            print("Эквализация выполнена ->", out)

        elif args.mode == "gamma":
            result = gamma_correct(img, args.gamma)
            out = args.output or "out_gamma.png"
            _save_image(out, result)
            print("Гамма-коррекция (gamma=%.2f) -> %s" % (args.gamma, out))

        else:  # histogram
            hist = compute_histogram(img, args.bins)
            out = args.output or "histogram.txt"
            with open(out, "w", encoding="utf-8") as f:
                if isinstance(hist, list):
                    for ch, h in enumerate(hist):
                        f.write("# channel %d\n" % ch)
                        for i, val in enumerate(h):
                            f.write("%d\t%.0f\n" % (i, val[0]))
                else:
                    for i, val in enumerate(hist):
                        f.write("%d\t%.0f\n" % (i, val[0]))
            print("Гистограмма (%d bins) -> %s" % (args.bins, out))

    except (IOError, ValueError) as e:
        print("[ERROR]", e)

def _read_wav(path):
    try:
        with wave.open(path, "rb") as w:
            params = w.getparams()
            frames = w.readframes(params.nframes)
    except FileNotFoundError:
        raise FileNotFoundError("Файл %r не найден" % path)
    except wave.Error as e:
        raise ValueError("Не .wav или повреждённый файл: %s" % e)

    if params.sampwidth != 2:
        raise ValueError("Поддерживается только 16-битный PCM (sampwidth=2)")

    import numpy as np
    samples = np.frombuffer(frames, dtype=np.int16)
    return samples, params


def _save_wav(path, samples, params):
    try:
        with wave.open(path, "wb") as w:
            w.setparams(params._replace(nframes=len(samples)))
            w.writeframes(samples.tobytes())
    except (OSError, IOError, wave.Error) as e:
        raise IOError("Не удалось сохранить %r: %s" % (path, e))


def _histogram_256(samples):
    import numpy as np
    hist, edges = np.histogram(samples, bins=256, range=(-32768, 32768))
    return hist, edges


def _quantize(samples, bins=256):
    import numpy as np
    if bins < 2:
        raise ValueError("bins должно быть >= 2")
    step = 65536.0 / bins
    idx = np.floor((samples.astype(np.float64) + 32768.0) / step)
    idx = np.clip(idx, 0, bins - 1)
    restored = (idx + 0.5) * step - 32768.0
    return np.clip(restored, -32768, 32767).astype(np.int16)


def run_wav_section(argv):
    p = argparse.ArgumentParser(
        prog="ht3.py wav",
        description="Чтение .wav, гистограмма 256 бинов, квантование.")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="quantized.wav")
    p.add_argument("--bins", type=int, default=256)
    p.add_argument("--hist-out", default="wav_histogram.txt")
    args = p.parse_args(argv)

    try:
        samples, params = _read_wav(args.input)
    except (FileNotFoundError, ValueError) as e:
        print("[ERROR]", e)
        return

    hist, edges = _histogram_256(samples)
    try:
        with open(args.hist_out, "w", encoding="utf-8") as f:
            for i in range(len(hist)):
                f.write("%.0f\t%d\n" % (edges[i], hist[i]))
        print("Гистограмма ->", args.hist_out)
    except (OSError, IOError) as e:
        print("[ERROR] не удалось записать гистограмму:", e)

    quant = _quantize(samples, args.bins)
    try:
        _save_wav(args.output, quant, params)
        print("Квантованный .wav ->", args.output)
    except IOError as e:
        print("[ERROR]", e)


def main():
    p = argparse.ArgumentParser(
        description="HT_3: матрицы + изображения + .wav")
    p.add_argument("section", choices=["matrix", "image", "wav"])
    args, rest = p.parse_known_args()

    if args.section == "matrix":
        run_matrix_section()
    elif args.section == "image":
        run_image_section(rest)
    elif args.section == "wav":
        run_wav_section(rest)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
