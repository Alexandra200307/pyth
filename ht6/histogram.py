import copy

import decoders.decoder
import encoders.encoder


class Hist:
    @classmethod
    def read(cls, file_path):
        ext = file_path.rsplit('.', 1)[-1].lower()

        if ext == 'bin':
            decoder = decoders.decoder.BinHistDecoder
        elif ext == 'txt':
            decoder = decoders.decoder.TxtHistDecoder
        elif ext == 'json':
            decoder = decoders.decoder.JsonHistDecoder
        elif ext == 'csv':
            decoder = decoders.decoder.CsvHistDecoder
        elif ext in ('png', 'jpg', 'jpeg', 'bmp'):
            decoder = decoders.decoder.ImageHistDecoder
        else:
            raise RuntimeError('Невозможно получить данные %s' % file_path)

        data = decoder.decode(file_path)
        return cls(data)

    @classmethod
    def write(cls, file_path, data):
        ext = file_path.rsplit('.', 1)[-1].lower()

        if ext == 'bin':
            encoder = encoders.encoder.BinHistEncoder
        elif ext == 'txt':
            encoder = encoders.encoder.TxtHistEncoder
        elif ext == 'json':
            encoder = encoders.encoder.JsonHistEncoder
        elif ext == 'csv':
            encoder = encoders.encoder.CsvHistEncoder
        elif ext in ('png', 'jpg', 'jpeg', 'bmp'):
            encoder = encoders.encoder.ImageHistEncoder
        else:
            raise RuntimeError('Невозможно записать данные %s' % file_path)

        encoder.encode(file_path, data)

    def __init__(self, data):
        self._data = data

    def get_data(self):
        return copy.deepcopy(self._data)

    def save(self, file_path):
        self.write(file_path, self._data)


if __name__ == "__main__":
    hist = Hist.read('./data/csv_test.csv')
    print(hist.get_data())

    # Пример записи
    hist.save('./data/csv_test_out.csv')
    hist.save('./data/csv_test_out.json')