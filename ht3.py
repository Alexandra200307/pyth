import cv2
import numpy as np
import argparse

def read_image(filename):
    ext = filename.split('.')[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']:
        raise ValueError("Unsupported image format")
    data = np.fromfile(filename, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def write_image(filename, img):
    ext = '.' + filename.split('.')[-1].lower()
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(filename)

def compute_histogram(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = np.zeros(256, dtype=int)
    for row in gray:
        for val in row:
            hist[val] += 1
    return hist

def equalize_histogram(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    hist = compute_histogram(gray)
    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    cdf_min = 0
    for v in cdf:
        if v > 0:
            cdf_min = v
            break
    total = gray.shape[0] * gray.shape[1]
    result = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            val = gray[i, j]
            if cdf[val] > cdf_min:
                result[i, j] = int((cdf[val] - cdf_min) * 255 / (total - cdf_min))
            else:
                result[i, j] = 0
    return result

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    result = np.zeros_like(image)
    if len(image.shape) == 3:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(3):
                    result[i, j, k] = min(255, int((image[i, j, k] / 255.0) ** inv_gamma * 255))
    else:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = min(255, int((image[i, j] / 255.0) ** inv_gamma * 255))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to image file")
    parser.add_argument("mode", choices=["hist", "equalize", "gamma"], help="Mode")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma value")
    args = parser.parse_args()

    img = read_image(args.filename)
    if img is None:
        print("Error reading image")
        return

    if args.mode == "hist":
        hist = compute_histogram(img)
        print(hist)
    elif args.mode == "equalize":
        result = equalize_histogram(img)
        write_image("equalized.jpg", result)
    elif args.mode == "gamma":
        result = gamma_correction(img, args.gamma)
        write_image("gamma_corrected.jpg", result)

if __name__ == "__main__":
    main()