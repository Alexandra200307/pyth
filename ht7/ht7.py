import numpy as np
from PIL import Image
from functools import partial
from pathlib import Path

# 2. Конвертация в оттенки серого через лямбда-функцию и map
def to_grayscale(img):
    pixels = np.array(img).reshape(-1, 3)
    get_gray = lambda p: int(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2])
    return np.array(list(map(get_gray, pixels)), dtype=np.float64).reshape(img.height, img.width)

# 1. Итераторы обхода
def iter_linear(h, w):
    return ((y, x) for y in range(h) for x in range(w))

def iter_spiral(h, w):
    y, x = h // 2, w // 2
    yield y, x
    count = 1
    step = 1
    while count < h * w:
        for _ in range(step):
            x += 1
            if 0 <= x < w and 0 <= y < h:
                yield y, x
                count += 1
        for _ in range(step):
            y += 1
            if 0 <= x < w and 0 <= y < h:
                yield y, x
                count += 1
        step += 1
        for _ in range(step):
            x -= 1
            if 0 <= x < w and 0 <= y < h:
                yield y, x
                count += 1
        for _ in range(step):
            y -= 1
            if 0 <= x < w and 0 <= y < h:
                yield y, x
                count += 1
        step += 1

def iter_zigzag(h, w):
    for s in range(h + w - 1):
        if s % 2 == 0:
            for y in range(s, -1, -1):
                x = s - y
                if 0 <= y < h and 0 <= x < w: yield y, x
        else:
            for x in range(s, -1, -1):
                y = s - x
                if 0 <= y < h and 0 <= x < w: yield y, x

def iter_peano_hilbert(h, w):
    n = 1
    while n < max(h, w): n *= 2
    for d in range(n * n):
        x, y = 0, 0
        t = d
        s = 1
        while s < n:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        if x < w and y < h:
            yield y, x

# 3. Сверточный оператор (ленивые вычисления)
def lazy_convolve(img, kernel, coords):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = img.shape
    
    for y, x in coords:
        if ph <= y < h - ph and pw <= x < w - pw:
            patch = img[y-ph:y+ph+1, x-pw:x+pw+1]
            yield y, x, np.sum(patch * kernel)

def get_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

# 4. Тестирование и MSE
def apply_filter(img, coord_iterator, filter_func):
    out = np.zeros_like(img)
    for y, x, val in filter_func(img, coords=coord_iterator(*img.shape)):
        out[y, x] = val
    return out

def calc_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

if __name__ == "__main__":
    target_dir = Path(r"C:\Users\Пользователь\AppData\Local\Programs\Microsoft VS Code")
    img_path = target_dir / "image.png"
    
    if not img_path.exists():
        img_path = target_dir / "image.jpg"
        
    try:
        img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Файл не найден в {target_dir}")
        exit()

    gray_img = to_grayscale(img)

    # Фиксация параметров фильтра Гаусса через partial
    gauss_kernel = get_gaussian_kernel(5, 1.5)
    filter_partial = partial(lazy_convolve, kernel=gauss_kernel)

    # Применение фильтра с разными обходами
    res_linear = apply_filter(gray_img, iter_linear, filter_partial)
    res_spiral = apply_filter(gray_img, iter_spiral, filter_partial)
    res_zigzag = apply_filter(gray_img, iter_zigzag, filter_partial)
    res_peano = apply_filter(gray_img, iter_peano_hilbert, filter_partial)

    print("Сравнение результатов (MSE):")
    print(f"Linear vs Spiral: {calc_mse(res_linear, res_spiral):.5f}")
    print(f"Linear vs Zigzag: {calc_mse(res_linear, res_zigzag):.5f}")
    print(f"Linear vs Peano:  {calc_mse(res_linear, res_peano):.5f}")
