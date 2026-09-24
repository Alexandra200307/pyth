import os, time, random
from funcs import (matmul, matvec, trace, dot, histogram, # type: ignore
                   kernel_filter, read_data, write_data)


def measure(func):

    t0 = time.time()
    func()
    time_ = time.time() - t0
    return time_


def vec(n):
    return [random.uniform(-1, 1) for _ in range(n)]


def mat(n):
    return [vec(n) for _ in range(n)]


if __name__ == "__main__":
    results = []

    def add(name, size, t):
        results.append((name, size, t))
        print(f"{name:14s} {size:>8}  {t:.6f} s")

    sizes = [10, 25, 50, 100, 150]
    for n in sizes:
        A, B = mat(n), mat(n)
        add("matmul", n, measure(lambda: matmul(A, B)))

    for n in [50, 100, 200, 400]:
        A, v = mat(n), vec(n)
        add("matvec", n, measure(lambda: matvec(A, v)))

    for n in sizes:
        A = mat(n)
        add("trace", n, measure(lambda: trace(A)))

    for n in [100, 1000, 10000, 100000]:
        a, b = vec(n), vec(n)
        add("dot", n, measure(lambda: dot(a, b)))

    for n in [100, 1000, 10000, 100000]:
        d = [random.random() for _ in range(n)]
        for bins in [8, 16, 32, 64, 128]:
            add(f"hist({bins})", n, measure(lambda: histogram(d, bins)))

    k = [-1, 0, 1]
    for n in [100, 1000, 10000, 100000]:
        d = vec(n)
        add("kernel", n, measure(lambda: kernel_filter(d, k)))

    for n in [100, 1000, 10000, 100000]:
        d = vec(n)
        fn = f"t{n}.json"
        add("write", n, measure(lambda: write_data(fn, d)))
        add("read",  n, measure(lambda: read_data(fn)))
        os.remove(fn)

    out = os.path.join(os.path.expanduser("~"), "Desktop", "results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"{'function':<14} {'size':>10} {'time_sec':>14}\n")
        f.write("-" * 40 + "\n")
        for name, size, t in results:
            f.write(f"{name:<14} {size:>10} {t:>14.8f}\n")