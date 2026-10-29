import csv
import json
import struct
import cv2
import numpy as np

class HistEncoder:
    @staticmethod
    def encode(file_path, data):
        raise NotImplementedError()

class BinHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        values = [float(data.get(i, 0.0)) for i in range(256)]
        payload = struct.pack('256f', *values)
        with open(file_path, 'wb') as f:
            f.write(payload)

class CsvHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for key in sorted(data.keys()):
                writer.writerow([int(key), float(data[key])])

class TxtHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        with open(file_path, 'w') as f:
            for key in sorted(data.keys()):
                f.write(f"{int(key)} {float(data[key])}\n")

class JsonHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        keys = sorted(data.keys())
        payload = {
            'keys': [int(k) for k in keys],
            'values': [float(data[k]) for k in keys],
        }
        with open(file_path, 'w') as f:
            json.dump(payload, f)

class ImageHistEncoder(HistEncoder):
    @staticmethod
    def encode(file_path, data):
        size = 256
        total_pixels = size * size
        values = np.array([float(data.get(i, 0.0)) for i in range(256)], dtype=np.float64)
        values = np.clip(values, 0.0, None)
        
        s = values.sum()
        if s > 0:
            values = values / s
            
        counts = np.round(values * total_pixels).astype(int)
        diff = total_pixels - int(counts.sum())
        if diff != 0:
            idx = int(counts.argmax())
            counts[idx] = max(0, counts[idx] + diff)
            
        pixels = np.repeat(np.arange(256, dtype=np.uint8), counts)
        if pixels.size < total_pixels:
            pixels = np.concatenate([pixels, np.zeros(total_pixels - pixels.size, dtype=np.uint8)])
        elif pixels.size > total_pixels:
            pixels = pixels[:total_pixels]
            
        image = pixels.reshape((size, size))
        cv2.imwrite(file_path, image)