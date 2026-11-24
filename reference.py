import numpy as np

import config
from detector import corrected_image
from registration import align, make_anchors


class ReferenceStore:
    """Хранение эталонного кадра в памяти (ТЗ 6)."""

    def __init__(self):
        self._reference = None

    def set(self, frame):
        self._reference = frame.copy()

    def get(self):
        return self._reference

    def has(self) -> bool:
        return self._reference is not None

    def clear(self):
        self._reference = None


def build_reference(frames, zone=None):
    """Медиана кадров, совмещённых с первым пригодным кадром."""
    if not frames:
        return None
    required = len(frames) // 2 + 1
    for index, base in enumerate(frames):
        anchors = make_anchors(base)
        if not align(base, base, anchors, zone).usable:
            continue
        aligned = [base]
        for frame_index, frame in enumerate(frames):
            if frame_index == index:
                continue
            result = align(base, frame, anchors, zone)
            if result.usable:
                image = corrected_image(result)
                image[~result.valid] = base[~result.valid]
                aligned.append(image)
        if len(aligned) >= required:
            stack = np.stack(aligned)
            median = np.median(stack, axis=0).astype(np.uint8)
            # Блики и неполностью совмещённые кромки между кадрами не
            # должны менять контур базового кадра при записи эталона.
            unstable = np.ptp(stack, axis=0) > config.PIXEL_THRESHOLD
            if unstable.ndim == 3:
                unstable = unstable.any(axis=2)
            median[unstable] = base[unstable]
            return median
    return None
