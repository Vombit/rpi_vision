import cv2
import numpy as np

import config
import registration


def corrected_image(alignment):
    brightness = alignment.brightness
    if alignment.image.ndim == 3:
        brightness = brightness[..., None]
    return np.uint8(np.clip(np.rint(alignment.image - brightness), 0, 255))


def _distance_to_neighborhood(base, image, window):
    return cv2.max(cv2.subtract(cv2.erode(image, window), base),
                   cv2.subtract(base, cv2.dilate(image, window)))


def _mutual_difference(reference, observed, diameter):
    window = np.ones((diameter, diameter), np.uint8)
    return cv2.min(_distance_to_neighborhood(reference, observed, window),
                   _distance_to_neighborhood(observed, reference, window))


def _difference(reference, observed, alignment, zone, broad_highlights):
    if zone and alignment.defect_only:
        # Положение зоны неизвестно. Изменение доказано, только если нужная
        # яркость отсутствует во всём допустимом диапазоне смещения.
        diameter = 2 * config.REGISTRATION_MAX_SHIFT + 1
        window = np.ones((diameter, diameter), np.uint8)
        observed = registration.gray(observed)
        base = registration.gray(reference)
        return _distance_to_neighborhood(base, observed, window)
    # При клиппинге сдвиг блика тоже оставляет тонкие полосы вдоль кромок.
    if broad_highlights:
        wide = _mutual_difference(reference, observed, 9)
        # Контрастная вставка может быть узкой. Для сильной разницы
        # сохраняем меньший допуск на смещение кромки.
        near = _mutual_difference(reference, observed, 5)
        diff = cv2.max(wide, cv2.subtract(near, 40))
    elif alignment.scaled:
        # После изменения масштаба блик и кромка могут расходиться на пару
        # рабочих пикселей. Протяжённое изменение остаётся в обеих проверках.
        diff = _mutual_difference(reference, observed, 5)
    else:
        diff = cv2.absdiff(reference, observed)
    return diff.max(axis=2) if diff.ndim == 3 else diff


def _same_texture(reference, observed):
    """Ищет ту же локальную текстуру при другой яркости или контрасте."""
    reference = registration.gray(reference).astype(np.float32)
    observed = registration.gray(observed).astype(np.float32)
    height, width = reference.shape
    window = (13, 13)
    mean_reference = cv2.blur(reference, window)
    var_reference = cv2.blur(reference * reference, window) - mean_reference * mean_reference
    textured = var_reference >= 36
    same = np.zeros(reference.shape, dtype=bool)
    offsets = ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2),
               (-1, -1), (1, 1), (-1, 1), (1, -1))
    for dx, dy in offsets:
        if dx or dy:
            transform = np.float32([[1, 0, dx], [0, 1, dy]])
            local = cv2.warpAffine(observed, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)
        else:
            local = observed
        mean_local = cv2.blur(local, window)
        var_local = cv2.blur(local * local, window) - mean_local * mean_local
        covariance = cv2.blur(reference * local, window) - mean_reference * mean_local
        same |= (textured & (var_local >= 36) &
                 (covariance > 0.65 * np.sqrt(np.maximum(0, var_reference * var_local))))
    return same


def compare(reference, current, anchors=None, zone=None) -> dict:
    if anchors is None:
        anchors = registration.make_anchors(reference)
    alignment = registration.align(reference, current, anchors, zone)
    broad_highlights = np.mean(registration.gray(reference) >= 248) >= 0.15
    observed = corrected_image(alignment)
    diff = _difference(reference, observed, alignment, zone, broad_highlights)

    valid = alignment.valid.copy()
    if alignment.scaled:
        valid = cv2.erode(np.uint8(valid), np.ones((5, 5), np.uint8)).astype(bool)
    zone_transform = None
    offset = (0, 0)
    minimum_area = config.MIN_AREA
    if zone:
        left, top, right, bottom = zone
        valid[:top] = valid[bottom:] = False
        valid[:, :left] = valid[:, right:] = False
        zone_area = (right - left) * (bottom - top)
        minimum_area = max(9, round(config.MIN_AREA * zone_area / valid.size))
        if alignment.layers and alignment.usable:
            zone_transform = alignment.layers[0][0]
            center = np.float32([(left + right) / 2, (top + bottom) / 2])
            offset = tuple(zone_transform[:, :2] @ center + zone_transform[:, 2] - center)

    diff[~valid] = 0
    if broad_highlights and alignment.usable and np.any(diff > config.PIXEL_THRESHOLD):
        diff[_same_texture(reference, alignment.image)] = 0
    binary = np.uint8((diff > config.PIXEL_THRESHOLD) & (alignment.usable or alignment.defect_only)) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= minimum_area
    keep[0] = False
    filtered = np.uint8(keep[labels]) * 255
    if broad_highlights and keep.any():
        # Восстанавливаем площадь вокруг подтверждённого ядра. Тонкие
        # полосы без ядра по-прежнему не участвуют в пороге полного кадра.
        neighborhood = cv2.dilate(filtered, np.ones((5, 5), np.uint8)) != 0
        residual = cv2.absdiff(reference, observed)
        if residual.ndim == 3:
            residual = residual.max(axis=2)
        growth_threshold = max(config.PIXEL_THRESHOLD, 90) if zone else config.PIXEL_THRESHOLD
        expanded = neighborhood & valid & (residual > growth_threshold)
        filtered[expanded] = 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= minimum_area
        keep[0] = False
        filtered = np.uint8(keep[labels]) * 255
        diff[expanded] = residual[expanded]
    reference_boxes = []
    boxes = []
    components = []
    height, width = reference.shape[:2]
    for i in range(1, count):
        if not keep[i]:
            continue
        x, y, w, h, area = (int(value) for value in stats[i])
        reference_boxes.append((x, y, w, h, area))
        region = np.s_[y:y + h, x:x + w]
        component = labels[region] == i
        components.append(component)
        transform, _ = max(alignment.layers, key=lambda item: np.count_nonzero(item[1][region] & component))
        left, top, right, bottom = registration.transform_bounds((x, y, x + w, y + h), transform)
        x0, y0 = max(0, round(left)), max(0, round(top))
        x1, y1 = min(width, round(right)), min(height, round(bottom))
        boxes.append((x0, y0, x1 - x0, y1 - y0, area))

    changed_pixels = int(stats[keep, cv2.CC_STAT_AREA].sum())
    denominator = zone_area if zone else np.count_nonzero(valid)
    if alignment.defect_only and not zone:
        denominator = max(denominator, config.REGISTRATION_MIN_COVERAGE * valid.size)
    change_ratio = changed_pixels / denominator if denominator else 0.0
    changed = bool(changed_pixels and change_ratio >= config.CHANGE_RATIO_THRESHOLD)
    usable = alignment.usable or changed
    return {
        "usable": usable,
        "unchecked_reason": None if usable else alignment.issue,
        "clipping": alignment.clipping,
        "changed": changed,
        "change_ratio": change_ratio,
        "changed_pixels": changed_pixels,
        "reference_boxes": reference_boxes,
        "components": components,
        "boxes": boxes,
        "offset": offset,
        "zone_transform": zone_transform,
        "diff": diff,
        "mask": filtered,
        "valid": np.uint8(valid) * 255,
    }
