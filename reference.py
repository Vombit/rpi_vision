import numpy as np


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


def build_reference(frames) -> np.ndarray:
    """Медианное объединение нескольких кадров — шум не попадает в эталон."""
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)
