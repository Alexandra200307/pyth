import os
import cv2
import image
import object_analysis
from histogram import Hist

# Чтобы пути работали корректно при запуске из VS Code
os.chdir(os.path.dirname(os.path.abspath(__file__)))

img_path = '1.jpg'
csv_path = 'csv_test.csv'

print("1. Тест фабрики изображений:")
for i in range(3):
    try:
        reader = image.get_image_reader(i)
        img = reader.read_image(img_path)
        print(f"  Reader {i}: {img.shape if img is not None else 'Ошибка'}")
    except Exception as e:
        print(f"  Reader {i}: {e}")

print("\n2. Тест Object Analysis:")
img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

bin_obj = object_analysis.BinaryImage()
mono_obj = object_analysis.MonochromeImage()
color_obj = object_analysis.ColorImage()
filt_obj = object_analysis.FilteredAnalysis(bin_obj)

print(f"  Binary: {len(bin_obj.template_method(img_gray)[4])} шт.")
print(f"  Monochrome: {len(mono_obj.template_method(img_gray)[4])} шт.")
print(f"  Color: {len(color_obj.template_method(img_color)[4])} шт.")

# Проверка декоратора и Hu-моментов
x, y, w, h, area, hu = filt_obj.template_method(img_gray)
print(f"  Filtered: {len(area)} шт.")
if len(area) > 0:
    print(f"  Первый объект: x={x[0]}, y={y[0]}, area={area[0]}")
    print(f"  Hu-моменты: {[round(val, 5) for val in hu[0][:3]]}")

print("\n3. Тест гистограмм:")
hist = Hist.read(csv_path)
val_100 = hist.get_data().get(100, 0.0)
print(f"  Исходное значение [100]: {val_100:.6e}")

formats = ['bin', 'txt', 'json', 'csv', 'png']
for fmt in formats:
    out_file = f"out.{fmt}"
    try:
        hist.save(out_file)
        new_val = Hist.read(out_file).get_data().get(100, 0.0)
        diff = abs(val_100 - new_val)
        print(f"  {fmt.upper()}: разница = {diff:.6e}")
    except Exception as e:
        print(f"  {fmt.upper()}: Ошибка ({e})")