import cv2
import numpy as np

import config
from preprocessing import normalize_brightness


def _merge_boxes(boxes, gap=config.BLUR_KERNEL):
    """Сливает рамки, которые пересекаются или почти касаются (зазор <= gap).

    Компоненты сами по себе не пересекаются, но после масштабирования/по соседству
    удобнее показать одну общую рамку. Площадь — сумма площадей исходных регионов.
    """
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            if changed:
                break
            x1, y1, w1, h1, a1 = merged[i]
            for j in range(i + 1, len(merged)):
                x2, y2, w2, h2, a2 = merged[j]
                if (
                    x1 - gap <= x2 + w2
                    and x2 - gap <= x1 + w1
                    and y1 - gap <= y2 + h2
                    and y2 - gap <= y1 + h1
                ):
                    nx, ny = min(x1, x2), min(y1, y2)
                    merged[i] = [
                        nx,
                        ny,
                        max(x1 + w1, x2 + w2) - nx,
                        max(y1 + h1, y2 + h2) - ny,
                        a1 + a2,
                    ]
                    del merged[j]
                    changed = True
                    break
    return [tuple(b) for b in merged]


def compare(reference, current) -> dict:
    """Сравнение рабочих (уже предобработанных) кадров.

    Возвращает changed / change_ratio / changed_pixels и debug-изображения
    (diff, mask) для отладочного режима (ТЗ 8-10, 17).
    """
    current = normalize_brightness(current, reference)

    diff = cv2.absdiff(reference, current)
    _, mask = cv2.threshold(diff, config.PIXEL_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Удаление мелких connected components (< MIN_AREA)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    boxes = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= config.MIN_AREA:
            filtered[labels == i] = 255
            boxes.append(
                (
                    int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_HEIGHT]),
                    int(stats[i, cv2.CC_STAT_AREA]),
                )
            )
    boxes = _merge_boxes(boxes)

    changed_pixels = int(np.count_nonzero(filtered))
    change_ratio = changed_pixels / filtered.size
    changed = change_ratio >= config.CHANGE_RATIO_THRESHOLD and changed_pixels >= config.MIN_AREA

    return {
        "changed": changed,
        "change_ratio": change_ratio,
        "changed_pixels": changed_pixels,
        "boxes": boxes,
        "diff": diff,
        "mask": filtered,
    }
