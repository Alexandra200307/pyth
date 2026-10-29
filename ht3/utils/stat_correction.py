import math
import numpy as np

def processing(hist, image):
    av1 = np.array([key * value for key, value in hist.items()]).sum()
    std1 = math.sqrt(np.array([value * (key - av1) ** 2 for key, value in hist.items()]).sum())

    av2 = image.mean()
    std2 = image.std()
    
    if std2 == 0:
        std2 = 1.0
        
    image = std1 * (image - av2) / std2 + av1

    image = np.clip(image, 0, 255).astype('uint8')
    
    return image
