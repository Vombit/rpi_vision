import time
from queue import Empty, SimpleQueue
from threading import Lock

import cv2
from PySide6.QtCore import QThread, Signal

import camera as camera_module
import config


class CaptureThread(QThread):
    """Читает свою камеру и хранит последний полный кадр до передачи в обработку."""

    camera_error = Signal(object)
    frame_captured = Signal(object)
    settings_read = Signal(object, str, object)
    settings_applied = Signal(object, object, object, object, object)

    def __init__(self, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._frame_lock = Lock()
        self._latest = None
        self._settings_requests = SimpleQueue()

    def take_frame(self):
        with self._frame_lock:
            frame, self._latest = self._latest, None
            return frame

    def request_settings(self, token):
        self._settings_requests.put((token, None))

    def request_settings_apply(self, token, values):
        self._settings_requests.put((token, dict(values)))

    def _handle_settings(self, cam):
        try:
            token, requested = self._settings_requests.get_nowait()
        except Empty:
            return
        try:
            if requested is None:
                self.settings_read.emit(token, *cam.read_opencv_controls())
            else:
                rejected, unverified, error = cam.apply_opencv_controls(self._camera_index, requested)
                _, actual = cam.read_opencv_controls()
                self.settings_applied.emit(token, actual, rejected, unverified, error)
        except (camera_module.CameraError, cv2.error):
            if requested is None:
                self.settings_read.emit(token, "", {})
            else:
                self.settings_applied.emit(token, {}, dict.fromkeys(requested), [], None)

    def run(self):
        cam = camera_module.Camera()
        if not cam.open(self._camera_index):
            self.camera_error.emit(self)
            return
        max_fps = float(getattr(config, "CAPTURE_MAX_FPS", 30))
        min_interval_ms = 1000.0 / max_fps if max_fps > 0 else 0.0
        while not self.isInterruptionRequested():
            self._handle_settings(cam)
            started = time.perf_counter()
            try:
                frame = cam.read()
            except camera_module.CameraError:
                self.camera_error.emit(self)
                break
            if frame is not None and frame.size:
                with self._frame_lock:
                    notify = self._latest is None
                    self._latest = (started, frame)
                if notify:
                    self.frame_captured.emit(self)
            # Только ограничение сверху: если камера быстрее max_fps —
            # досыпаем до минимального интервала, если медленнее —
            # сразу идём за следующим кадром.
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            sleep_ms = int(min_interval_ms - elapsed_ms)
            if sleep_ms >= 1:
                self.msleep(sleep_ms)
        cam.close()

    def stop(self):
        self.requestInterruption()
        self.quit()
