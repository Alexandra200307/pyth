import sys
import struct
import csv
import json
import cv2

class HistDecoder:
    @staticmethod
    def decode(file_path):
        raise NotImplementedError()


class BinHistDecoder(HistDecoder):
    @staticmethod
    def decode(file_path):
        with open(file_path, 'rb') as file:
            raw_data = file.read(4 * 256)
        data = struct.unpack('256f', raw_data)
        return {i: data[i] for i in range(len(data))}

class CsvHistDecoder(HistDecoder):
    @staticmethod
    def decode(file_path):
        data = None
        with open(file_path) as csv_file:
            reader = csv.reader(csv_file)
            data = dict(reader)
            data = {int(i): float(key) for i, key in data.items()}
        return data

class TxtHistDecoder(HistDecoder):
    @staticmethod
    def decode(file_path):
        data = None
        with open(file_path, "r") as file:
            source = file.read().split()
            data = {int(i): float(source[2 * i + 1]) for i in range(len(source) // 2)}
        return data

class JsonHistDecoder(HistDecoder):
    @staticmethod
    def decode(file_path):
        data = None
        with open(file_path, 'r') as file:
            data = json.load(file)
        data = {data['keys'][i]: data['values'][i] for i in range(len(data['keys']))}
        return data

class ImageHistDecoder(HistDecoder):
    @staticmethod
    def decode(file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        num = image.shape[0] * image.shape[1]
        data = {i: (image == i).sum() / num for i in range(0, 256)}
        return data