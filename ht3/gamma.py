import numpy as np
import cv2

def processing(image, g=1.5):
    inv = 1.0 / g
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype('uint8')
    return cv2.LUT(image, table)
