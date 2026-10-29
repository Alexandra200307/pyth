def image_processing(image):
    num = image.shape[0] * image.shape[1]
    hist = {i: (image == i).sum() / num for i in range(0, 256)}
    return hist
