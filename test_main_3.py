import json
import os
import shutil
import tempfile

import numpy as np

from main import (
    matmul, matvec, trace, dot, histogram, kernel_filter,
    read_json, read_csv, read_txt, read_data, write_data,
    load_image, save_image, compute_histogram, equalize_histogram,
    gamma_correction,
)


TMP = tempfile.mkdtemp()


def p(name):
    return os.path.join(TMP, name)


def write(f, text):
    with open(f, "w", encoding="utf-8") as fp:
        fp.write(text)


def read(f):
    with open(f, encoding="utf-8") as fp:
        return fp.read()


def test_matmul():
    assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
    assert matmul([[1, 2, 3], [4, 5, 6]],
                  [[7, 8], [9, 10], [11, 12]]) == [[58, 64], [139, 154]]
    print("matmul ok")


def test_matvec():
    assert matvec([[1, 2], [3, 4]], [5, 6]) == [17, 39]
    assert matvec([[1, 0], [0, 1]], [7, 8]) == [7, 8]
    print("matvec ok")


def test_trace():
    assert trace([[1, 2], [3, 4]]) == 5
    assert trace([[1, 0, 0], [0, 2, 0], [0, 0, 3]]) == 6
    assert trace([[5]]) == 5
    print("trace ok")


def test_dot():
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert dot([0, 0], [1, 1]) == 0
    assert dot([-1, 2], [3, -4]) == -11
    print("dot ok")


def test_histogram():
    assert histogram([1, 2, 2, 3, 3, 3], bins=3) == [1, 2, 3]
    assert histogram([5, 5, 5], bins=2) == [3, 0]
    assert histogram([], bins=4) == [0, 0, 0, 0]
    print("histogram ok")


def test_kernel_filter():
    data = [1, 2, 3, 4, 5]
    assert kernel_filter(data, [1, 1, 1]) == [3.0, 6.0, 9.0, 12.0, 9.0]
    assert kernel_filter(data, [0, 1, 0]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    print("kernel_filter ok")


def test_read_json():
    f = p("data.json")
    write(f, json.dumps([1.5, 2.5, 3.5]))
    assert read_json(f) == [1.5, 2.5, 3.5]

    write(f, json.dumps([[1, 2], [3, 4]]))
    assert read_json(f) == [[1.0, 2.0], [3.0, 4.0]]
    print("read_json ok")


def test_read_csv():
    f = p("data.csv")
    write(f, "1.1,2.2,3.3\n")
    assert read_csv(f) == [1.1, 2.2, 3.3]

    write(f, "1\n2\n3\n")
    assert read_csv(f) == [1.0, 2.0, 3.0]

    write(f, "1,2\n3,4\n")
    assert read_csv(f) == [[1.0, 2.0], [3.0, 4.0]]
    print("read_csv ok")


def test_read_txt():
    f = p("data.txt")
    write(f, "1.5\n2.5\n3.5\n")
    assert read_txt(f) == [1.5, 2.5, 3.5]

    write(f, "1 2 3\n4 5 6\n")
    assert read_txt(f) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    print("read_txt ok")


def test_read_data():
    f = p("a.json")
    write(f, json.dumps([1, 2, 3]))
    assert read_data(f) == [1.0, 2.0, 3.0]

    f = p("b.csv")
    write(f, "4,5,6\n")
    assert read_data(f) == [4.0, 5.0, 6.0]

    f = p("c.txt")
    write(f, "7 8 9\n")
    assert read_data(f) == [[7.0, 8.0, 9.0]]

    f = p("d.dat")
    write(f, "10 11 12\n")
    assert read_data(f) == [[10.0, 11.0, 12.0]]
    print("read_data ok")


def test_write_data():
    f = p("out.json")
    write_data(f, [1.1, 2.2, 3.3])
    assert json.loads(read(f)) == [1.1, 2.2, 3.3]

    f = p("out.csv")
    write_data(f, [1.0, 2.0, 3.0])
    assert read(f).strip() == "1.0\n2.0\n3.0"

    f = p("out_mat.csv")
    write_data(f, [[1, 2], [3, 4]])
    assert read(f).strip() == "1,2\n3,4"

    f = p("out.txt")
    write_data(f, [4.0, 5.0])
    assert read(f).strip() == "4.0\n5.0"

    f = p("out_mat.txt")
    write_data(f, [[7, 8], [9, 10]])
    assert read(f).strip() == "7 8\n9 10"
    print("write_data ok")


def test_image_io():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:5, 2:5] = [255, 0, 0]

    f = p("img.png")
    save_image(f, img)
    assert os.path.exists(f)

    loaded = load_image(f)
    assert loaded is not None
    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)

    try:
        load_image(p("no_such_file.png"))
        assert False, "ожидалось FileNotFoundError"
    except FileNotFoundError:
        pass

    print("image_io ok")


def test_compute_histogram():
    gray = np.zeros((5, 5), dtype=np.uint8)
    gray[0, 0] = 255
    h = compute_histogram(gray, bins=256)
    assert len(h) == 256
    assert h[0] == 24
    assert h[255] == 1

    color = np.zeros((5, 5, 3), dtype=np.uint8)
    color[:, :, 0] = 100
    color[:, :, 1] = 150
    color[:, :, 2] = 200
    hs = compute_histogram(color, bins=256)
    assert len(hs) == 3
    assert hs[0][100] == 25
    assert hs[1][150] == 25
    assert hs[2][200] == 25
    print("compute_histogram ok")


def test_equalize_histogram():
    gray = np.array([
        [40, 40, 50, 50],
        [40, 50, 60, 60],
        [50, 60, 70, 70],
        [60, 70, 80, 80],
    ], dtype=np.uint8)
    eq = equalize_histogram(gray)
    assert eq.shape == gray.shape
    assert not np.array_equal(eq, gray)

    color = np.zeros((10, 10, 3), dtype=np.uint8)
    base = np.arange(100).reshape(10, 10)
    color[:, :, 0] = (base * 1) % 256
    color[:, :, 1] = (base * 2) % 256
    color[:, :, 2] = (base * 3) % 256
    eq_color = equalize_histogram(color)
    assert eq_color.shape == color.shape
    assert not np.array_equal(eq_color, color)
    print("equalize_histogram ok")


def test_gamma_correction():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)

    assert np.array_equal(gamma_correction(img, 1.0), img)
    assert gamma_correction(img, 2.0).mean() > img.mean()
    assert gamma_correction(img, 0.5).mean() < img.mean()

    for g in (0, -1.0):
        try:
            gamma_correction(img, g)
            assert False, "ожидалось ValueError"
        except ValueError:
            pass

    print("gamma_correction ok")


if __name__ == "__main__":
    tests = [
        test_matmul, test_matvec, test_trace, test_dot,
        test_histogram, test_kernel_filter,
        test_read_json, test_read_csv, test_read_txt,
        test_read_data, test_write_data,
        test_image_io, test_compute_histogram,
        test_equalize_histogram, test_gamma_correction,
    ]
    for t in tests:
        t()
    shutil.rmtree(TMP)