import cv2
import numpy as np


class ObjectAnalysis(object):
    def template_method(self, image):
        image = self.noise_filtering(image)
        data = self.segmentation(image)
        data = self.object_parameters(data)
        return data

    def noise_filtering(self, image):
        raise NotImplementedError()

    def segmentation(self, data):
        raise NotImplementedError()

    def object_parameters(self, data):
        (image, data) = data
        (numLabels, labels, stats, centroids) = data

        x = []
        y = []
        w = []
        h = []
        area = []
        hu = []

        for i in range(1, numLabels):
            x.append(stats[i, cv2.CC_STAT_LEFT])
            y.append(stats[i, cv2.CC_STAT_TOP])
            w.append(stats[i, cv2.CC_STAT_WIDTH])
            h.append(stats[i, cv2.CC_STAT_HEIGHT])
            area.append(stats[i, cv2.CC_STAT_AREA])

            mask = (labels == i).astype(np.uint8)
            moments = cv2.moments(mask)
            hu_moments = cv2.HuMoments(moments).flatten().tolist()
            hu.append(hu_moments)

        return (x, y, w, h, area, hu)


class BinaryImage(ObjectAnalysis):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        return cv2.medianBlur(image, 5)

    def segmentation(self, image):
        output = cv2.connectedComponentsWithStats(
            image,
            4,  # connectivity
            cv2.CV_32S
        )
        return (image, output)


class MonochromeImage(BinaryImage):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def segmentation(self, image):
        edges = cv2.Canny(image, 100, 200)

        output = cv2.connectedComponentsWithStats(
            edges,
            8,
            cv2.CV_32S
        )
        return (edges, output)


class ColorImage(MonochromeImage):
    def __init__(self):
        pass

    def noise_filtering(self, image):
        return image

    def segmentation(self, image):
        if image is None:
            raise ValueError("Изображение не загружено")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            watershed_image = image
        else:
            gray = image
            watershed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(
            thresh, cv2.MORPH_OPEN, kernel, iterations=2
        )

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        if dist.max() > 0:
            _, sure_fg = cv2.threshold(
                dist, 0.7 * dist.max(), 255, 0
            )
        else:
            sure_fg = np.zeros_like(dist, dtype=np.uint8)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(watershed_image, markers)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers > 1] = 255

        output = cv2.connectedComponentsWithStats(
            mask,
            8,
            cv2.CV_32S
        )
        return (image, output)


class FilteredAnalysis(ObjectAnalysis):
    """
    Декоратор: фильтрует объекты по площади и сохраняет Hu-моменты.
    """

    def __init__(self, obj):
        self._proc = obj

    def template_method(self, image):
        (_x, _y, _w, _h, _area, _hu) = self._proc.template_method(image)

        x = []
        y = []
        w = []
        h = []
        area = []
        hu = []

        for i in range(len(_area)):
            if 10 < _area[i] < 2500:
                x.append(_x[i])
                y.append(_y[i])
                w.append(_w[i])
                h.append(_h[i])
                area.append(_area[i])
                hu.append(_hu[i])

        return (x, y, w, h, area, hu)


if __name__ == '__main__':
    image_path = './data/1.jpg'

    print("Binary Image Processing")
    bin_segm = BinaryImage()
    x, y, w, h, area, hu = bin_segm.template_method(
        cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    )
    for i in range(len(area)):
        print([x[i], y[i], w[i], h[i], area[i], hu[i]])

    print("Decorated Binary Image Processing")
    filt_bin = FilteredAnalysis(BinaryImage())
    x, y, w, h, area, hu = filt_bin.template_method(
        cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    )
    for i in range(len(area)):
        print([x[i], y[i], w[i], h[i], area[i], hu[i]])

    print("Monochrome Image Processing")
    mono = MonochromeImage()
    x, y, w, h, area, hu = mono.template_method(
        cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    )
    for i in range(len(area)):
        print([x[i], y[i], w[i], h[i], area[i], hu[i]])

    print("Color Image Processing")
    color = ColorImage()
    x, y, w, h, area, hu = color.template_method(
        cv2.imread(image_path, cv2.IMREAD_COLOR)
    )
    for i in range(len(area)):
        print([x[i], y[i], w[i], h[i], area[i], hu[i]])