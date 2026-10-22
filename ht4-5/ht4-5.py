import datetime
import traceback


class LogLevel:
    DEBUG = "DEBUG"
    INFO  = "INFO"
    WARN  = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Logger:
    def __init__(self, filename="Log.txt", user=None):
        self.filename = filename
        self.user = user if user is not None else self._new_session_id()

        try:
            with open(self.filename, "a", encoding="utf-8"):
                pass
        except (OSError, IOError) as e:
            raise RuntimeError("Не удалось открыть файл логов %r: %s"
                               % (self.filename, e))

        self.info("Logger", "Сессия логгера создана", event_id="LOG-INIT")

    @staticmethod
    def _new_session_id():
        return datetime.datetime.now().strftime("sess-%Y%m%d-%H%M%S-%f")

    def _write_line(self, level, source, message, event_id, extra):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = " | ".join([
            ts,
            level,
            str(source),
            str(event_id) if event_id is not None else "-",
            str(self.user),
            str(message),
            str(extra) if extra is not None else "-",
        ])

        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except (OSError, IOError) as e:
            print("[LOGGER ERROR] %s" % e)
        finally:
            pass

    def log(self, level, source, message, event_id=None, extra=None):
        allowed = (LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN,
                   LogLevel.ERROR, LogLevel.FATAL)
        if level not in allowed:
            raise ValueError("Недопустимый уровень логирования: %r" % (level,))
        self._write_line(level, source, message, event_id, extra)

    def debug(self, source, message, event_id=None, extra=None):
        self.log(LogLevel.DEBUG, source, message, event_id, extra)

    def info(self, source, message, event_id=None, extra=None):
        self.log(LogLevel.INFO, source, message, event_id, extra)

    def warn(self, source, message, event_id=None, extra=None):
        self.log(LogLevel.WARN, source, message, event_id, extra)

    def error(self, source, message, event_id=None, extra=None):
        self.log(LogLevel.ERROR, source, message, event_id, extra)

    def fatal(self, source, message, event_id=None, extra=None):
        self.log(LogLevel.FATAL, source, message, event_id, extra)

    def exception(self, source, message, event_id=None, extra=None):
        """Логирует текущее исключение с трассировкой стека."""
        tb = traceback.format_exc()
        full_extra = (str(extra) + "\n" if extra else "") + tb
        self.error(source, message, event_id, extra=full_extra)

class PersonError(Exception):
    pass


def make_patronymic(father_name, gender):
    if not isinstance(father_name, str):
        raise PersonError("father_name должен быть строкой")
    if gender not in ("male", "female"):
        raise PersonError("gender должен быть 'male' или 'female'")

    if not father_name:
        return ""

    if gender == "male":
        if father_name.endswith(("а", "я")):
            return father_name[:-1] + "ич"
        if father_name.endswith(("й", "ь")):
            return father_name[:-1] + "евич"
        return father_name + "ович"
    else:
        if father_name.endswith(("а", "я")):
            return father_name[:-1] + "ична"
        if father_name.endswith(("й", "ь")):
            return father_name[:-1] + "евна"
        return father_name + "овна"


class Person:
    def __init__(self, first_name, last_name, gender, birth_year,
                 middle_name=None, father=None, mother=None, branch=None,
                 logger=None):
        if not isinstance(first_name, str) or not first_name.strip():
            raise PersonError("Имя должно быть непустой строкой")
        if not isinstance(last_name, str) or not last_name.strip():
            raise PersonError("Фамилия должна быть непустой строкой")
        if gender not in ("male", "female"):
            raise PersonError("Неверный пол: %r" % (gender,))
        if not isinstance(birth_year, int) or not (0 <= birth_year <= 2100):
            raise PersonError("Год рождения должен быть целым 0..2100")
        if father is not None and not isinstance(father, Person):
            raise PersonError("father должен быть Person или None")
        if mother is not None and not isinstance(mother, Person):
            raise PersonError("mother должен быть Person или None")

        self.first_name = first_name
        self.last_name = last_name
        self.gender = gender
        self.birth_year = birth_year
        self.father = father
        self.mother = mother
        self.spouse = None
        self.children = []
        self.branch = branch
        self.logger = logger

        if middle_name is not None:
            self.middle_name = middle_name
        elif branch is not None:
            self.middle_name = ""     
        elif father is not None:
            self.middle_name = make_patronymic(father.first_name, gender)
        else:
            self.middle_name = ""

        if father is not None:
            father.children.append(self)
        if mother is not None:
            mother.children.append(self)

        if self.logger is not None:
            self.logger.info(
                "Person", "Создан Person: %s %s %s"
                % (self.last_name, self.first_name, self.middle_name),
                event_id="PERSON-CREATE"
            )

    def __repr__(self):
        return "Person(%s %s %s)" % (
            self.last_name, self.first_name, self.middle_name
        )


class Family:
    def __init__(self, surname, address=None, branch=None, logger=None):
        if not isinstance(surname, str) or not surname.strip():
            raise PersonError("Фамилия семьи должна быть непустой строкой")

        self.surname = surname
        self.partners = []
        self.children = []
        self.address = address
        self.branch = branch
        self.logger = logger

    def add_partner(self, person):
        if not isinstance(person, Person):
            raise PersonError("В семью можно добавить только Person")
        if person in self.partners:
            return person
        self.partners.append(person)
        person.last_name = self.surname
        if self.logger is not None:
            self.logger.debug("Family", "Партнёр добавлен: %r" % (person,),
                              event_id="FAMILY-ADD-PARTNER")
        return person

    def add_child(self, child):
        if not isinstance(child, Person):
            raise PersonError("Ребёнком может быть только Person")
        if child in self.children:
            return child
        self.children.append(child)
        child.last_name = self.surname
        if self.logger is not None:
            self.logger.debug("Family", "Ребёнок добавлен: %r" % (child,),
                              event_id="FAMILY-ADD-CHILD")
        return child

    def all_members(self):
        return self.partners + self.children


class House:
    def __init__(self, address, capacity=10, logger=None):
        if not isinstance(address, str) or not address.strip():
            raise PersonError("Адрес дома должен быть непустой строкой")
        if not isinstance(capacity, int) or capacity <= 0:
            raise PersonError("capacity должен быть положительным целым")

        self.address = address
        self.capacity = capacity
        self.families = []
        self.logger = logger

    def add_family(self, family):
        if not isinstance(family, Family):
            raise PersonError("В дом можно добавить только Family")
        if len(self.families) >= self.capacity:
            msg = "Дом %s переполнен (%d/%d)" % (
                self.address, len(self.families), self.capacity
            )
            if self.logger is not None:
                self.logger.error("House", msg, event_id="HOUSE-OVERFLOW")
            raise PersonError(msg)

        self.families.append(family)
        family.address = self.address
        if self.logger is not None:
            self.logger.info("House", "Семья %s заселена в %s"
                             % (family.surname, self.address),
                             event_id="HOUSE-ADD-FAMILY")


class City:
    def __init__(self, name, logger=None):
        if not isinstance(name, str) or not name.strip():
            raise PersonError("Название города должно быть непустой строкой")
        self.name = name
        self.houses = {}
        self.logger = logger

    def add_house(self, address, capacity=10):
        if address in self.houses:
            if self.logger is not None:
                self.logger.warn("City", "Дом %s уже существует" % address,
                                 event_id="CITY-HOUSE-EXISTS")
            return self.houses[address]
        house = House(address, capacity, logger=self.logger)
        self.houses[address] = house
        return house

    def place_family(self, family, address):
        if not isinstance(family, Family):
            raise PersonError("place_family ожидает Family")
        if not isinstance(address, str) or not address.strip():
            raise PersonError("Адрес должен быть непустой строкой")

        try:
            if address not in self.houses:
                self.add_house(address)
            self.houses[address].add_family(family)
        except PersonError:
            if self.logger is not None:
                self.logger.exception(
                    "City", "Ошибка заселения семьи %s" % family.surname,
                    event_id="CITY-PLACE-FAIL"
                )
            raise
        finally:
            if self.logger is not None:
                self.logger.debug(
                    "City", "Попытка заселения семьи %s в %s завершена"
                    % (family.surname, address),
                    event_id="CITY-PLACE-DONE"
                )

    def find_families(self, surname):
        if not isinstance(surname, str):
            raise PersonError("surname должен быть строкой")
        result = []
        for house in self.houses.values():
            for family in house.families:
                if family.surname == surname:
                    result.append(family)
        return result

class ImageError(Exception):
    pass


class ImageIOError(ImageError):
    pass


def clip(value):
    return max(0, min(255, int(value)))


class Image:

    MAGIC = "IMG"

    def __init__(self, width, height, pixels=None, logger=None):
        if not isinstance(width, int) or width <= 0:
            raise ImageError("width должен быть положительным целым")
        if not isinstance(height, int) or height <= 0:
            raise ImageError("height должен быть положительным целым")

        self.width = width
        self.height = height
        self.logger = logger

        if pixels is None:
            self.pixels = [[0] * width for _ in range(height)]
        else:
            if not isinstance(pixels, list) or len(pixels) != height:
                raise ImageError(
                    "pixels: ожидалось %d строк, получено %r"
                    % (height, len(pixels) if isinstance(pixels, list) else type(pixels))
                )
            for y, row in enumerate(pixels):
                if not isinstance(row, list) or len(row) != width:
                    raise ImageError(
                        "pixels[%d]: ожидалось %d элементов" % (y, width)
                    )
            self.pixels = [row[:] for row in pixels]

    def copy(self):
        return type(self)(self.width, self.height, self.pixels,
                          logger=self.logger)

    def get(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ImageError("Выход за границы изображения: (%r, %r)" % (x, y))
        return self.pixels[y][x]

    def set(self, x, y, value):
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ImageError("Выход за границы изображения: (%r, %r)" % (x, y))
        self.pixels[y][x] = value

    def _serialize_pixel(self, v):
        raise NotImplementedError

    def save(self, path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("%s %s\n" % (self.MAGIC, type(self).__name__))
                f.write("%d %d\n" % (self.width, self.height))
                for row in self.pixels:
                    f.write(" ".join(self._serialize_pixel(v) for v in row))
                    f.write("\n")
        except (OSError, IOError) as e:
            raise ImageIOError("Ошибка записи файла %r: %s" % (path, e))
        finally:
            if self.logger is not None:
                self.logger.debug("ImageIO", "save(%r) завершён" % path,
                                  event_id="IMG-SAVE")

    @classmethod
    def load(cls, path, logger=None):
        try:
            f = open(path, "r", encoding="utf-8")
        except FileNotFoundError:
            raise ImageIOError("Файл %r не найден" % path)
        except IsADirectoryError:
            raise ImageIOError("%r — это директория, а не файл" % path)
        except (OSError, IOError) as e:
            raise ImageIOError("Не удалось открыть %r: %s" % (path, e))

        try:
            with f:
                header = f.readline().strip().split()
                if len(header) != 2 or header[0] != cls.MAGIC:
                    raise ImageIOError("Неверный заголовок файла: %r" % header)

                kind = header[1]
                if kind != cls.__name__:
                    raise ImageIOError(
                        "Файл содержит %s, а ожидался %s" % (kind, cls.__name__)
                    )

                wh = f.readline().split()
                if len(wh) != 2:
                    raise ImageIOError("Неверная строка размеров")
                try:
                    width, height = int(wh[0]), int(wh[1])
                except ValueError:
                    raise ImageIOError("Размеры должны быть целыми: %r" % wh)

                pixels = []
                for y in range(height):
                    line = f.readline()
                    if not line:
                        raise ImageIOError(
                            "Неожиданный конец файла (строка %d)" % (y + 2)
                        )
                    parts = line.strip().split()
                    if len(parts) != width:
                        raise ImageIOError(
                            "Строка %d: ожидалось %d пикселей, получено %d"
                            % (y + 2, width, len(parts))
                        )
                    pixels.append([cls._parse_pixel_static(p) for p in parts])

                return cls(width, height, pixels, logger=logger)
        finally:
            if logger is not None:
                logger.debug("ImageIO", "load(%r) завершён" % path,
                             event_id="IMG-LOAD")

    @staticmethod
    def _parse_pixel_static(s):
        raise NotImplementedError


class BinaryImage(Image):
    def __init__(self, width, height, pixels=None, logger=None):
        Image.__init__(self, width, height, pixels, logger=logger)
        for y, row in enumerate(self.pixels):
            for x, v in enumerate(row):
                if v not in (0, 1):
                    raise ImageError(
                        "BinaryImage: пиксель (%d,%d)=%r должен быть 0 или 1"
                        % (x, y, v)
                    )

    def _serialize_pixel(self, v):
        return str(int(v))

    @staticmethod
    def _parse_pixel_static(s):
        try:
            return int(s)
        except ValueError:
            raise ImageIOError("Не число в бинарном изображении: %r" % s)


class MonochromeImage(Image):
    def __init__(self, width, height, pixels=None, logger=None):
        Image.__init__(self, width, height, pixels, logger=logger)
        for y, row in enumerate(self.pixels):
            for x, v in enumerate(row):
                if not isinstance(v, int) or not (0 <= v <= 255):
                    raise ImageError(
                        "MonochromeImage: пиксель (%d,%d)=%r не в 0..255"
                        % (x, y, v)
                    )

    def _serialize_pixel(self, v):
        return str(int(v))

    @staticmethod
    def _parse_pixel_static(s):
        try:
            v = int(s)
        except ValueError:
            raise ImageIOError("Не число в монохроме: %r" % s)
        if not (0 <= v <= 255):
            raise ImageIOError("Значение вне 0..255: %r" % v)
        return v


class ColorImage(Image):
    def __init__(self, width, height, pixels=None, logger=None):
        Image.__init__(self, width, height, pixels, logger=logger)
        for y, row in enumerate(self.pixels):
            for x, v in enumerate(row):
                if not (isinstance(v, (tuple, list)) and len(v) == 3
                        and all(isinstance(c, int) and 0 <= c <= 255 for c in v)):
                    raise ImageError(
                        "ColorImage: пиксель (%d,%d)=%r не RGB 0..255"
                        % (x, y, v)
                    )

    def _serialize_pixel(self, v):
        return "%d,%d,%d" % (v[0], v[1], v[2])

    @staticmethod
    def _parse_pixel_static(s):
        parts = s.split(",")
        if len(parts) != 3:
            raise ImageIOError("Ожидалось R,G,B, получено %r" % s)
        try:
            r, g, b = (int(p) for p in parts)
        except ValueError:
            raise ImageIOError("Не числа в RGB: %r" % s)
        for c in (r, g, b):
            if not (0 <= c <= 255):
                raise ImageIOError("Компонента вне 0..255: %r" % c)
        return (r, g, b)

class ImageConverter:

    def __init__(self, logger=None):
        self.logger = logger

    @staticmethod
    def _stretch_map(vals, low_percent=1, high_percent=99):
        if not vals:
            return lambda v: v
        sorted_vals = sorted(vals)
        n = len(sorted_vals)
        lo_idx = max(0, min(n - 1, int(low_percent / 100.0 * (n - 1))))
        hi_idx = max(0, min(n - 1, int(high_percent / 100.0 * (n - 1))))
        lo, hi = sorted_vals[lo_idx], sorted_vals[hi_idx]
        if hi <= lo:
            return lambda v: v
        return lambda v: clip(round((v - lo) * 255.0 / (hi - lo)))

    def _check_type(self, img, cls, what):
        if not isinstance(img, cls):
            raise ImageError("%s: ожидался %s, получен %s"
                             % (what, cls.__name__, type(img).__name__))
        if len(img.pixels) != img.height or \
           any(len(r) != img.width for r in img.pixels):
            raise ImageError("%s: повреждены данные (несоответствие размеров)"
                             % what)

    def mono_to_mono_stat(self, img, low_percent=1, high_percent=99):
        try:
            self._check_type(img, MonochromeImage, "mono_to_mono_stat")
            vals = [v for row in img.pixels for v in row]
            f = self._stretch_map(vals, low_percent, high_percent)
            out = [[f(v) for v in row] for row in img.pixels]
            return MonochromeImage(img.width, img.height, out,
                                   logger=self.logger)
        except ImageError:
            if self.logger:
                self.logger.exception("ImageConverter", "mono_to_mono_stat",
                                      event_id="CVT-A")
            raise
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "mono_to_mono_stat завершено",
                                  event_id="CVT-A-DONE")

    def color_to_color_stat(self, img, low_percent=1, high_percent=99):
        try:
            self._check_type(img, ColorImage, "color_to_color_stat")
            r_vals, g_vals, b_vals = [], [], []
            for row in img.pixels:
                for r, g, b in row:
                    r_vals.append(r); g_vals.append(g); b_vals.append(b)
            fr = self._stretch_map(r_vals, low_percent, high_percent)
            fg = self._stretch_map(g_vals, low_percent, high_percent)
            fb = self._stretch_map(b_vals, low_percent, high_percent)
            out = [[(fr(r), fg(g), fb(b)) for r, g, b in row]
                   for row in img.pixels]
            return ColorImage(img.width, img.height, out, logger=self.logger)
        except ImageError:
            if self.logger:
                self.logger.exception("ImageConverter", "color_to_color_stat",
                                      event_id="CVT-B")
            raise
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "color_to_color_stat завершено",
                                  event_id="CVT-B-DONE")

    def binary_to_binary(self, img):
        try:
            self._check_type(img, BinaryImage, "binary_to_binary")
            return img.copy()
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "binary_to_binary завершено",
                                  event_id="CVT-C")

    def color_to_mono(self, img):
        try:
            self._check_type(img, ColorImage, "color_to_mono")
            out = [[(r + g + b) // 3 for r, g, b in row]
                   for row in img.pixels]
            return MonochromeImage(img.width, img.height, out, logger=self.logger)
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "color_to_mono завершено",
                                  event_id="CVT-D")

    def mono_to_color(self, img, palette):
        try:
            self._check_type(img, MonochromeImage, "mono_to_color")
            if not isinstance(palette, (list, tuple)) or len(palette) != 256:
                raise ImageError("Палитра должна содержать ровно 256 цветов")
            for i, c in enumerate(palette):
                if not (isinstance(c, (tuple, list)) and len(c) == 3
                        and all(isinstance(x, int) and 0 <= x <= 255 for x in c)):
                    raise ImageError("palette[%d]=%r не RGB 0..255" % (i, c))
            out = [[tuple(palette[v]) for v in row] for row in img.pixels]
            return ColorImage(img.width, img.height, out, logger=self.logger)
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "mono_to_color завершено",
                                  event_id="CVT-E")

    @staticmethod
    def _otsu_threshold(img):
        hist = [0] * 256
        for row in img.pixels:
            for v in row:
                hist[v] += 1
        total = img.width * img.height
        sum_total = sum(i * hist[i] for i in range(256))
        sum_b = 0.0; w_b = 0; max_var = 0.0; threshold = 0
        for t in range(256):
            w_b += hist[t]
            if w_b == 0: continue
            w_f = total - w_b
            if w_f == 0: break
            sum_b += t * hist[t]
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var = w_b * w_f * (m_b - m_f) ** 2
            if var > max_var:
                max_var = var; threshold = t
        return threshold

    def mono_to_binary(self, img, threshold=None):
        try:
            self._check_type(img, MonochromeImage, "mono_to_binary")
            if threshold is None:
                threshold = self._otsu_threshold(img)
            if not isinstance(threshold, int) or not (0 <= threshold <= 255):
                raise ImageError("threshold должен быть 0..255")
            out = [[1 if v > threshold else 0 for v in row]
                   for row in img.pixels]
            return BinaryImage(img.width, img.height, out, logger=self.logger)
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "mono_to_binary завершено",
                                  event_id="CVT-F")

    def binary_to_mono_distance(self, img):
        try:
            self._check_type(img, BinaryImage, "binary_to_mono_distance")
            w, h = img.width, img.height
            INF = 10 ** 9
            dist = [[INF] * w for _ in range(h)]
            queue = []; head = 0

            for y in range(h):
                for x in range(w):
                    if img.pixels[y][x] > 0:
                        dist[y][x] = 0
                        queue.append((x, y))

            if not queue:
                return MonochromeImage(w, h, [[0] * w for _ in range(h)],
                                       logger=self.logger)

            dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
            while head < len(queue):
                x, y = queue[head]; head += 1
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and dist[ny][nx] == INF:
                        dist[ny][nx] = dist[y][x] + 1
                        queue.append((nx, ny))

            max_d = max(max(r) for r in dist)
            if max_d == 0:
                return MonochromeImage(w, h, [[0] * w for _ in range(h)],
                                       logger=self.logger)
            out = [[int(round(255.0 * dist[y][x] / max_d)) for x in range(w)]
                   for y in range(h)]
            return MonochromeImage(w, h, out, logger=self.logger)
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "binary_to_mono_distance завершено",
                                  event_id="CVT-G")

    def color_to_binary(self, img, threshold=None):
        try:
            mono = self.color_to_mono(img)
            return self.mono_to_binary(mono, threshold)
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "color_to_binary завершено",
                                  event_id="CVT-H")

    def binary_to_color(self, img, palette):
        try:
            mono = self.binary_to_mono_distance(img)
            return self.mono_to_color(mono, palette)
        finally:
            if self.logger:
                self.logger.debug("ImageConverter", "binary_to_color завершено",
                                  event_id="CVT-I")

if __name__ == "__main__":
    logger = Logger("Log.txt", user="demo-user")

    # --- Город ---
    city = City("Примерск", logger=logger)
    father = Person("Иван", "Петров", "male", 1980, logger=logger)
    mother = Person("Мария", "Петрова", "female", 1982, logger=logger)
    child  = Person("Алексей", "Петров", "male", 2010,
                    father=father, mother=mother, logger=logger)
    print("Отчество:", child.middle_name)

    family = Family("Петровы", logger=logger)
    family.add_partner(father)
    family.add_partner(mother)
    family.add_child(child)
    city.place_family(family, "ул. Ленина, 1")

    try:
        Person("Х", "Y", "cat", 2000, logger=logger)
    except PersonError as e:
        logger.exception("Person", "Ожидаемая ошибка валидации",
                         event_id="PERSON-BAD-GENDER")
        print("Поймали:", e)

    small = House("Малый", capacity=1, logger=logger)
    small.add_family(family)
    try:
        small.add_family(family)
    except PersonError as e:
        print("Переполнение:", e)

    conv = ImageConverter(logger=logger)

    mono = MonochromeImage(3, 2, [[0, 100, 200], [50, 150, 255]],
                           logger=logger)
    binary = conv.mono_to_binary(mono, threshold=127)
    print("Binary:", binary.pixels)

    try:
        mono.get(5, 5)
    except ImageError as e:
        logger.exception("Image", "Выход за границы", event_id="IMG-OOB")
        print("OOB:", e)

    try:
        conv.mono_to_color(mono, palette=[(0, 0, 0)] * 10)
    except ImageError as e:
        print("Палитра:", e)

    try:
        binary.save("binary.txt")
        loaded = BinaryImage.load("binary.txt", logger=logger)
        print("Загружено:", loaded.pixels)
    except ImageIOError as e:
        print("I/O:", e)

    try:
        BinaryImage.load("no_such_file.txt", logger=logger)
    except ImageIOError as e:
        print("Нет файла:", e)

    try:
        BinaryImage.load("binary.txt", logger=logger).save
        ColorImage.load("binary.txt", logger=logger)
    except ImageIOError as e:
        print("Не тот формат:", e)

    logger.info("Main", "Программа завершена", event_id="APP-END")