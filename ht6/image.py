import cv2


class AbstractFactoryImageReader:
    def read_image(self, file_path):
        raise NotImplementedError()


class BinImageReader(AbstractFactoryImageReader):
    def read_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(file_path)

        # Бинаризация: удобно для дальнейшего анализа связных компонент
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary


class MonochromeImageReader(AbstractFactoryImageReader):
    def read_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(file_path)
        return image


class ColorImageReader(AbstractFactoryImageReader):
    def read_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(file_path)
        return image


def get_image_reader(ident):
    if ident == 0:
        return BinImageReader()
    elif ident == 1:
        return MonochromeImageReader()
    elif ident == 2:
        return ColorImageReader()
    raise ValueError("Неизвестный идентификатор читателя изображения: %s" % ident)


if __name__ == "__main__":
    for i in range(3):
        print(get_image_reader(i))