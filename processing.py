import time
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal, Slot

import config
from controller import Confirmation, Controller

STATUS_NORMAL = "NORMAL"
STATUS_ALARM = "ALARM"
STATUS_NO_REFERENCE = "NO_REFERENCE"
STATUS_UNCHECKED = "UNCHECKED"


@dataclass
class FrameResult:
    generation: int
    frame: object
    roi: tuple | None
    status: str
    has_reference: bool = False
    elapsed_ms: float = 0.0
    change_ratio: float = 0.0
    boxes: list = field(default_factory=list)
    reference_pending: bool = False
    defect_pending: bool = False
    unchecked_reason: str | None = None
    clipping: tuple[float, float] | None = None


@dataclass
class DetectionRequest:
    started: float
    confirmation: Confirmation = field(default_factory=Confirmation)
    count: int = 0


class Processor(QObject):
    """Обрабатывает кадры и команды в одном рабочем потоке."""

    result_ready = Signal(object)
    detection_done = Signal(int, str)

    def __init__(self, controller: Controller, parent=None):
        super().__init__(parent)
        self._controller = controller
        self._generation = 0
        self._source = None
        self._roi = None
        self._reference_frames = None
        self._reference_started = 0.0
        self._reference_failed = False
        self._detection = None

    @Slot(int, object, object)
    def configure(self, generation, source, roi):
        self._generation = generation
        self._source = source
        self._roi = roi
        self._reference_frames = None
        self._reference_failed = False
        self._detection = None
        self._controller.clear_reference()

    @Slot()
    def reset_confirmation(self):
        self._controller.reset_confirmation()
        if self._detection is not None:
            self._detection = DetectionRequest(time.perf_counter())

    @Slot(float)
    def on_set_reference(self, started):
        self._controller.clear_reference()
        self._reference_frames = []
        self._reference_started = started
        self._reference_failed = False
        if self._detection is not None:
            self._finish_detection(STATUS_UNCHECKED)

    @Slot(float, int)
    def on_detect(self, started, generation):
        if generation != self._generation:
            return
        if not self._controller.has_reference():
            self._finish_detection(STATUS_UNCHECKED)
            return
        self._detection = DetectionRequest(started)

    def _finish_detection(self, status):
        self._detection = None
        self.detection_done.emit(self._generation, status)

    @Slot(object)
    def on_frame_available(self, source):
        if source is not self._source:
            return
        latest = source.take_frame()
        if latest is not None:
            self._process(*latest)

    def _process(self, captured_at, frame):
        if self._reference_frames is not None and captured_at >= self._reference_started:
            self._reference_frames.append(frame)
            if len(self._reference_frames) >= config.REFERENCE_FRAMES:
                self._reference_failed = not self._controller.set_reference(
                    self._reference_frames, roi=self._roi)
                self._reference_frames = None

        if not self._controller.has_reference():
            status = STATUS_UNCHECKED if self._reference_failed else STATUS_NO_REFERENCE
            self.result_ready.emit(FrameResult(
                self._generation, frame, self._roi, status,
                reference_pending=self._reference_frames is not None,
            ))
            return

        result = self._controller.check_frame(frame)
        status = self._status(result, result["confirmed"])
        self.result_ready.emit(FrameResult(
            self._generation, frame, result["preview_roi"], status, True,
            result["processing_time_ms"], result["change_ratio"],
            (result["confirmed_boxes"] or result["boxes"]) if result["usable"] else [],
            defect_pending=result["changed"] and not result["confirmed"],
            unchecked_reason=result["unchecked_reason"],
            clipping=result["clipping"],
        ))
        if (request := self._detection) is not None and captured_at >= request.started:
            request.count += 1
            confirmed = bool(request.confirmation.update(result))
            if not result["usable"] or request.count >= config.CONFIRMATION_FRAMES:
                self._finish_detection(self._status(result, confirmed))

    @staticmethod
    def _status(result, confirmed):
        if not result["usable"]:
            return STATUS_UNCHECKED
        if confirmed:
            return STATUS_ALARM
        return STATUS_UNCHECKED if result["changed"] else STATUS_NORMAL
