import cv2

import config


def roi_bounds(shape, roi):
    height, width = shape[:2]
    if roi is None:
        return 0, 0, width, height
    x0 = min(max(round(roi[0] * width), 0), width - 1)
    y0 = min(max(round(roi[1] * height), 0), height - 1)
    x1 = min(max(round(roi[2] * width), x0 + 1), width)
    y1 = min(max(round(roi[3] * height), y0 + 1), height)
    return x0, y0, x1, y1


def crop_roi(frame, roi):
    x0, y0, x1, y1 = roi_bounds(frame.shape, roi)
    return frame[y0:y1, x0:x1]


def preprocess(frame):
    out = cv2.resize(frame, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
                     interpolation=cv2.INTER_AREA)
    if config.USE_GRAYSCALE and out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if config.BLUR_KERNEL > 1:
        k = config.BLUR_KERNEL
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out
