import sys
import os
import argparse

from utils.reader import image_reader as imread
from utils.reader import csv_reader, bin_reader, txt_reader, json_reader
from utils.processor import histogram
from utils.writer import csv_writer, bin_writer, txt_writer, image_writer, json_writer
from utils.image_toner import stat_correction, equalize, gamma


IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
        return imread.read_data(path)
    elif ext == '.csv':
        return csv_reader.read_data(path)
    elif ext == '.bin':
        return bin_reader.read_data(path)
    elif ext == '.txt':
        return txt_reader.read_data(path)
    elif ext == '.json':
        return json_reader.read_data(path)
    else:
        raise ValueError(f"неизвестное расширение: {ext}")


def save_hist(path, hist):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        csv_writer.write_data(path, hist)
    elif ext == '.bin':
        bin_writer.write_data(path, hist)
    elif ext == '.txt':
        txt_writer.write_data(path, hist)
    elif ext == '.json':
        json_writer.write_data(path, hist)
    else:
        raise ValueError(f"не могу сохранить гистограмму с расширением: {ext}")


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--img_path', default='', help='Path to image')
    parser.add_argument('-p', '--path', default='', help='Path to template file')
    parser.add_argument('-o', '--output', default='', help='Save file path')
    parser.add_argument('-m', '--mode', default='hist',
                        choices=['hist', 'equalize', 'gamma', 'stat'],
                        help='Processing mode: hist / equalize / gamma / stat')
    return parser


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args(sys.argv[1:])

    if not args.img_path:
        print("нужно указать путь к изображению через -img")
        sys.exit(1)

    image = imread.read_data(args.img_path)
    if image is None:
        print("не удалось прочитать изображение")
        sys.exit(1)

    if args.mode == 'hist':
        hist = histogram.image_processing(image)
        print("гистограмма посчитана, ключей:", len(hist))
        if args.output:
            save_hist(args.output, hist)
            print("сохранено:", args.output)

    elif args.mode == 'equalize':
        res = equalize.processing(image)
        image_writer.write_data(args.output, res)
        print("сохранено после эквализации:", args.output)

    elif args.mode == 'gamma':
        res = gamma.processing(image, 1.5)
        image_writer.write_data(args.output, res)
        print("сохранено после гамма-коррекции:", args.output)

    elif args.mode == 'stat':
        if not args.path:
            print("для режима stat нужен файл-шаблон -p")
            sys.exit(1)
        template = read_any(args.path)
        if isinstance(template, dict):
            hist_template = template
        else:
            hist_template = histogram.image_processing(template)
        res = stat_correction.processing(hist_template, image)
        image_writer.write_data(args.output, res)
        print("сохранено после stat_correction:", args.output)
