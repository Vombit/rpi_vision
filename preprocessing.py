import cv2

import config


def preprocess(frame):
    """Приводит сырой BGR-кадр к рабочему виду: resize -> gray (опц.) -> blur."""
    out = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
                     interpolation=cv2.INTER_AREA)
    if config.USE_GRAYSCALE:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if config.BLUR_KERNEL > 1:
        k = config.BLUR_KERNEL
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def normalize_brightness(current, reference):
    """Компенсирует небольшой глобальный сдвиг яркости (ТЗ 11.2)."""
    delta = float(reference.mean()) - float(current.mean())
    if 0 < abs(delta) <= config.BRIGHTNESS_TOLERANCE:
        current = cv2.add(current, int(round(delta)))
    return current
