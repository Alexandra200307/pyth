import re
import numpy as np
from pathlib import Path
from PIL import Image

IMAGE_EXT_RE = re.compile(r'\.(?:png|jpe?g|bmp|gif|tiff?)$', re.IGNORECASE)

def is_allowed_file(filename: str) -> bool:
    return IMAGE_EXT_RE.search(filename) is not None

def load_and_encode_image(path: Path, p: int = 16, threshold: int = 128):
    if not is_allowed_file(path.name):
        raise ValueError(f"Недопустимый формат файла: {path.name}")

    img = Image.open(path).convert("L")
    arr = np.array(img)
    
    binary = (arr > threshold).astype(np.uint8)

    h, w = binary.shape
    rows, cols = h // p, w // p
    binary = binary[:rows * p, :cols * p]

    grid = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            patch = binary[i*p:(i+1)*p, j*p:(j+1)*p]
            grid[i, j] = int(patch.mean() >= 0.5)

    code = "".join(str(int(v)) for row in grid for v in row)
    return grid, code

def count_cross_regex(grid: np.ndarray) -> int:
    rows, cols = grid.shape
    if rows < 3 or cols < 3:
        return 0

    s = "\n".join("".join(str(int(v)) for v in row) for row in grid)
    total = 0

    for c in range(cols - 2):
        pattern = (
            "(?=101" + "." * (cols - c - 3) +
            "\n" + "." * c + "000" + "." * (cols - c - 3) +
            "\n" + "." * c + "101)"
        )
        total += sum(1 for _ in re.finditer(pattern, s))

    return total

if __name__ == "__main__":

    target_dir = Path(r"C:\Users\Пользователь\AppData\Local\Programs\Microsoft VS Code")
    img_path = target_dir / "image.png"
    
    grid, code = load_and_encode_image(img_path, p=16)
    
    print(f"r'{code}'")
    print(f"Длина строки: {len(code)}")
    print(f"Количество крестов: {count_cross_regex(grid)}")