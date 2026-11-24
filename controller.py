import logging
import time
from pathlib import Path

import cv2
import numpy as np

import camera as camera_module
import config
import detector
import preprocessing
import reference as reference_module
import registration

log = logging.getLogger(__name__)

REFERENCE_NOT_SET = "REFERENCE_NOT_SET"
CAMERA_ERROR = "CAMERA_ERROR"
_USE_CONFIGURED_ROI = object()


class Confirmation:
    def __init__(self):
        self._previous = []

    def update(self, result):
        candidates = zip(result["reference_boxes"], result["components"]) if result["changed"] else ()
        current = []
        confirmed = []
        for index, (box, mask) in enumerate(candidates):
            streak = 1 + max(
                (count for old, old_mask, count in self._previous if _overlaps(old, old_mask, box, mask)),
                default=0,
            )
            current.append((box, mask, streak))
            if streak >= config.CONFIRMATION_FRAMES:
                confirmed.append(result["boxes"][index])
        self._previous = current
        return confirmed


def _overlaps(first, first_mask, second, second_mask):
    x1, y1, w1, h1, area1 = first
    x2, y2, w2, h2, area2 = second
    left, top = max(x1, x2), max(y1, y2)
    right, bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if left >= right or top >= bottom:
        return False
    first_overlap = first_mask[top - y1:bottom - y1, left - x1:right - x1]
    second_overlap = second_mask[top - y2:bottom - y2, left - x2:right - x2]
    return np.count_nonzero(first_overlap & second_overlap) >= 0.30 * min(area1, area2)


class Controller:
    """Эталон, область проверки и подтверждение для одной камеры."""

    def __init__(self, get_roi=None):
        self._store = reference_module.ReferenceStore()
        self._get_roi = get_roi if get_roi is not None else lambda: None
        self._roi = None
        self._zone = None
        self._anchors = []
        self.reset_confirmation()

    def set_reference(self, frames=None, *, roi=_USE_CONFIGURED_ROI) -> bool:
        self.clear_reference()
        try:
            if frames is None:
                frames = [camera_module.capture_frame() for _ in range(config.REFERENCE_FRAMES)]
        except camera_module.CameraError as exc:
            log.error("Camera error while setting reference: %s", exc)
            return False

        processed = [preprocessing.preprocess(frame) for frame in frames]
        if not processed:
            return False
        self._roi = self._get_roi() if roi is _USE_CONFIGURED_ROI else roi
        if self._roi is not None:
            self._zone = preprocessing.roi_bounds(processed[0].shape, self._roi)
            if self._zone == (0, 0, processed[0].shape[1], processed[0].shape[0]):
                self._zone = None
        reference = reference_module.build_reference(processed, self._zone)
        if reference is None:
            log.warning("Reference rejected: insufficient aligned frames")
            return False
        anchors = registration.make_anchors(reference)
        if not registration.align(reference, reference, anchors, self._zone).usable:
            log.warning("Reference rejected: insufficient stable features")
            return False
        self._store.set(reference)
        self._anchors = anchors
        if config.DEBUG:
            _save_debug("reference.png", reference)
        log.info("Reference created (%d frames)", len(processed))
        return True

    def check_frame(self, frame, confirmation=None) -> dict:
        if not self.has_reference():
            return {"usable": False, "changed": False, "confirmed": False, "error": REFERENCE_NOT_SET}

        start = time.perf_counter()
        processed = preprocessing.preprocess(frame)
        result = detector.compare(self._store.get(), processed, self._anchors, self._zone)
        confirmation = self._confirmation if confirmation is None else confirmation
        result["confirmed_boxes"] = confirmation.update(result)
        result["confirmed"] = bool(result["confirmed_boxes"])
        result["processing_time_ms"] = (time.perf_counter() - start) * 1000.0
        result["preview_roi"] = self._preview_roi(result.pop("zone_transform"))

        if config.DEBUG:
            _save_debug("current.png", processed)
            _save_debug("difference.png", result["diff"])
            _save_debug("mask.png", result["mask"])
            _save_debug("valid.png", result["valid"])
        for key in ("diff", "mask", "valid"):
            del result[key]
        return result

    def _preview_roi(self, transform):
        if self._roi is None or transform is None:
            return self._roi
        width, height = config.IMAGE_WIDTH, config.IMAGE_HEIGHT
        left, top, right, bottom = self._roi
        bounds = registration.transform_bounds((left * width, top * height, right * width, bottom * height),
                                                transform)
        return tuple(float(value / size) for value, size in zip(bounds, (width, height, width, height)))

    def detect_change(self, frame=None, frames=None) -> dict:
        if not self.has_reference():
            return {"usable": False, "changed": False, "error": REFERENCE_NOT_SET}
        if frames is None:
            try:
                frames = [frame] if frame is not None else [
                    camera_module.capture_frame() for _ in range(config.CONFIRMATION_FRAMES)
                ]
            except camera_module.CameraError as exc:
                log.error("Camera error during detect_change: %s", exc)
                return {"usable": False, "changed": False, "error": CAMERA_ERROR}

        confirmation = Confirmation()
        result = {}
        elapsed = 0.0
        count = 0
        for current in frames:
            result = self.check_frame(current, confirmation)
            elapsed += result["processing_time_ms"]
            count += 1
            if not result["usable"] or count >= config.CONFIRMATION_FRAMES:
                break
        return {
            "usable": bool(result.get("usable") and count >= config.CONFIRMATION_FRAMES),
            "changed": bool(result.get("confirmed")),
            "change_ratio": result.get("change_ratio", 0.0),
            "changed_pixels": result.get("changed_pixels", 0),
            "boxes": result.get("confirmed_boxes", []),
            "processing_time_ms": elapsed,
        }

    def get_reference(self):
        return self._store.get()

    def has_reference(self) -> bool:
        return self._store.has()

    def clear_reference(self):
        self._store.clear()
        self._anchors = []
        self._roi = self._zone = None
        self.reset_confirmation()

    def reset_confirmation(self):
        self._confirmation = Confirmation()


def _save_debug(name, image):
    path = Path(config.DEBUG_DIR)
    path.mkdir(exist_ok=True)
    cv2.imwrite(str(path / name), image)
