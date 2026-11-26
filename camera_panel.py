import sys
import time

import cv2
from PySide6.QtCore import QSize, QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import config
import registration
import v4l2
from camera_settings import CameraSettingsDialog, OpenCVCameraSettingsDialog
from capture import CaptureThread
from controller import Controller
from preprocessing import roi_bounds
from processing import (
    STATUS_ALARM, STATUS_NO_REFERENCE, STATUS_NORMAL, STATUS_UNCHECKED,
    FrameResult, Processor,
)
from video_view import VideoView

NO_CAMERA = "NO_CAMERA"


class CameraPanel(QWidget):
    """Одна камера целиком: свой выбор камеры, свой захват, свой эталон,
    своя область интереса, свои кнопки и статусы.
    """

    status_changed = Signal(int, str)
    configuration_changed = Signal(int, object, object)
    reference_requested = Signal(float)
    detection_requested = Signal(float, int)
    confirmation_reset = Signal()

    def __init__(self, slot_index: int, preview_size, cameras, parent=None):
        super().__init__(parent)
        self.slot_index = slot_index
        self.roi = getattr(config, "DEFAULT_ROI", None)
        self._display_roi = None
        self._generation = 0
        self._status = NO_CAMERA
        self._capture_thread = None
        self._processor = Processor(Controller())
        self._process_thread = QThread(self)
        self._processor.moveToThread(self._process_thread)
        self._processor.result_ready.connect(self._on_result)
        self._processor.detection_done.connect(self._on_detection_done)
        self.configuration_changed.connect(self._processor.configure)
        self.reference_requested.connect(self._processor.on_set_reference)
        self.detection_requested.connect(self._processor.on_detect)
        self.confirmation_reset.connect(self._processor.reset_confirmation)
        self._process_thread.start()

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel(self.tr("Camera:")))
        self.camera_combo = QComboBox()
        self.camera_combo.activated.connect(self._on_camera_selected)
        self.settings_button = QPushButton(self.tr("Settings..."))
        self.settings_button.clicked.connect(self._on_settings)
        self._fill_combo(cameras)
        camera_row.addWidget(self.camera_combo, stretch=1)
        camera_row.addWidget(self.settings_button)
        column.addLayout(camera_row)

        self.video = VideoView()
        self.video.setMaximumSize(*preview_size)
        self.video.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet(
            "background-color: #202020; color: #aaaaaa; border: 1px solid #444;")
        self.video.setText(self.tr("Camera not available"))
        self.video.roi_selected.connect(self._on_roi_selected)
        # РАЗРЕШАЕМ РИСОВАТЬ РАМКУ ВСЕГДА
        self.video.set_selection_enabled(True)
        column.addWidget(self.video, stretch=1)

        transparent_bg = "background: transparent;"
        mono = "font-family: monospace; font-size: 14px; " + transparent_bg
        
        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-family: monospace; font-size: 14px; font-weight: bold; " + transparent_bg)
        self.reference_label = QLabel()
        self.reference_label.setStyleSheet(mono)
        self.change_label = QLabel()
        self.change_label.setStyleSheet(mono)
        self.last_check_label = QLabel()
        self.last_check_label.setStyleSheet(mono)
        self.trigger_label = QLabel()
        self.trigger_label.setStyleSheet(mono)

        for label in (self.status_label, self.reference_label, self.change_label,
                      self.last_check_label, self.trigger_label):
            column.addWidget(label)

        buttons_row = QHBoxLayout()
        self.reset_roi_button = QPushButton(self.tr("Reset zone"))
        self.reset_roi_button.setEnabled(False)
        self.reset_roi_button.clicked.connect(self._on_reset_roi)
        buttons_row.addWidget(self.reset_roi_button)
        
        self.set_reference_button = QPushButton(self.tr("Set reference"))
        self.set_reference_button.setEnabled(False)
        self.set_reference_button.clicked.connect(self._on_set_reference)
        buttons_row.addWidget(self.set_reference_button)
        
        column.addLayout(buttons_row)

        self._set_status(NO_CAMERA)
        self.change_label.setText(self.tr("Change: -"))
        self.trigger_label.setText(self.tr("Trigger: -"))

        self.start_capture(self.camera_combo.currentData())

    def _fill_combo(self, cameras):
        current = self.camera_combo.currentData()
        self.camera_combo.clear()
        for index in cameras:
            self.camera_combo.addItem(f"Camera {index}", userData=index)
        wanted = current
        if wanted is None or self.camera_combo.findData(wanted) < 0:
            wanted = cameras[self.slot_index] if self.slot_index < len(cameras) else None
        position = self.camera_combo.findData(wanted)
        self.camera_combo.setCurrentIndex(position if position >= 0 else -1)
        self.settings_button.setEnabled(self.camera_combo.currentData() is not None)

    def reload_cameras(self, cameras):
        self.stop_capture()
        self._fill_combo(cameras)
        if self.camera_combo.currentData() is None:
            self._on_camera_error()
            return False
        self._on_camera_selected()
        return True

    def _on_camera_selected(self):
        self.stop_capture()
        self.roi = getattr(config, "DEFAULT_ROI", None)
        self._display_roi = None
        self.reference_label.setText(self.tr("Reference: NOT SET"))
        self.trigger_label.setText(self.tr("Trigger: -"))
        self._set_status(NO_CAMERA)
        self.set_reference_button.setEnabled(False)
        self.reset_roi_button.setEnabled(False)
        self.start_capture(self.camera_combo.currentData())

    @Slot()
    def _on_settings(self):
        index = self.camera_combo.currentData()
        if index is None:
            return
        if sys.platform.startswith("linux") and v4l2.available():
            device = v4l2.device_for_index(index)
            dialog = CameraSettingsDialog(index, device, v4l2.controls_for_index(index), self)
        elif self._capture_thread and self._capture_thread.isRunning():
            dialog = OpenCVCameraSettingsDialog(self._capture_thread, self)
        else:
            QMessageBox.information(
                self, self.tr("Camera settings"),
                self.tr("Camera not available"))
            return
        dialog.applying.connect(self._on_camera_settings_changed)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    @Slot()
    def _on_camera_settings_changed(self):
        self._change_roi(self.roi)

    def start_capture(self, camera_index):
        if camera_index is None:
            self._on_camera_error()
            return
        self.stop_capture()
        self._capture_thread = CaptureThread(camera_index)
        self._capture_thread.camera_error.connect(self._on_capture_error)
        self._capture_thread.frame_captured.connect(self._processor.on_frame_available)
        self._capture_thread.finished.connect(self._capture_thread.deleteLater)
        self._configure_processor(self._capture_thread)
        self._capture_thread.start()

    def _configure_processor(self, source):
        self._generation += 1
        self.configuration_changed.emit(self._generation, source, self.roi)

    def stop_capture(self):
        thread, self._capture_thread = self._capture_thread, None
        if thread is not None:
            self._configure_processor(None)
            try:
                thread.camera_error.disconnect(self._on_capture_error)
            except Exception:
                pass
            try:
                thread.frame_captured.disconnect(self._processor.on_frame_available)
            except Exception:
                pass
            thread.stop()
            thread.wait(3000)

    def close_capture(self):
        self.stop_capture()
        self._process_thread.quit()
        self._process_thread.wait(3000)

    @Slot(float, float, float, float)
    def _on_roi_selected(self, x0: float, y0: float, x1: float, y1: float):
        left, top, right, bottom = self._display_roi or (0.0, 0.0, 1.0, 1.0)
        width, height = right - left, bottom - top
        self._change_roi((left + x0 * width, top + y0 * height,
                          left + x1 * width, top + y1 * height))

    def _change_roi(self, roi):
        self.roi = roi
        self._configure_processor(self._capture_thread)
        self.reference_label.setText(self.tr("Reference: NOT SET"))
        self.change_label.setText(self.tr("Change: -"))
        self.last_check_label.setText(self.tr("Last check: -"))
        self.trigger_label.setText(self.tr("Trigger: -"))
        self._set_status(STATUS_NO_REFERENCE)

    @Slot()
    def _on_set_reference(self):
        started = time.perf_counter()
        self._configure_processor(self._capture_thread)
        self._set_status(STATUS_NO_REFERENCE)
        self.reference_label.setText(self.tr("Reference: setting..."))
        self.trigger_label.setText(self.tr("Trigger: -"))
        self.reference_requested.emit(started)

    @Slot()
    def _on_reset_roi(self):
        self._change_roi(getattr(config, "DEFAULT_ROI", None))

    @Slot()
    def reset_roi(self):
        self._on_reset_roi()

    @Slot()
    def arm_reference(self):
        self._on_set_reference()

    @Slot()
    def request_detect(self):
        self.detection_requested.emit(time.perf_counter(), self._generation)

    @Slot(int, str)
    def _on_detection_done(self, generation, status):
        if generation != self._generation:
            return
        texts = {
            STATUS_ALARM: (self.tr("Trigger: CHANGED"), "#e33"),
            STATUS_NORMAL: (self.tr("Trigger: NO CHANGE"), "#3c3"),
            STATUS_UNCHECKED: (self.tr("Trigger: FRAME NOT CHECKED"), "#ca3"),
        }
        text, color = texts[status]
        self.trigger_label.setText(text)
        self.trigger_label.setStyleSheet(
            f"font-family: monospace; font-size: 14px; color: {color}; background: transparent;")

    @Slot(object)
    def _on_capture_error(self, source):
        if source is self._capture_thread:
            self._configure_processor(None)
            self._on_camera_error()

    def _on_camera_error(self):
        self.video.setPixmap(QPixmap())
        self.video.setText(self.tr("Camera not available"))
        self._set_status(NO_CAMERA)
        self.reference_label.setText(self.tr("Reference: NOT SET"))
        self.change_label.setText(self.tr("Change: -"))
        self.last_check_label.setText(self.tr("Last check: -"))
        self.trigger_label.setText(self.tr("Trigger: -"))
        self.settings_button.setEnabled(False)
        self.set_reference_button.setEnabled(False)
        self.reset_roi_button.setEnabled(False)

    @Slot(object)
    def _on_result(self, result: FrameResult):
        if result.generation != self._generation:
            return
        self._set_status(result.status, defect_pending=result.defect_pending)

        frame = result.frame
        height, width = frame.shape[:2]
        left, top, right, bottom = roi_bounds(frame.shape, result.roi)
        self._display_roi = (left / width, top / height, right / width, bottom / height)
        available = self.video.contentsRect().size().expandedTo(QSize(1, 1))
        size = QSize(right - left, bottom - top).scaled(
            available, Qt.AspectRatioMode.KeepAspectRatio)
        preview_width, preview_height = max(1, size.width()), max(1, size.height())
        preview = cv2.resize(frame[top:bottom, left:right], (preview_width, preview_height),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB if preview.ndim == 2 else cv2.COLOR_BGR2RGB)
        color = (255, 51, 51) if result.status == STATUS_ALARM else (255, 204, 51)
        sx, sy = preview_width / (right - left), preview_height / (bottom - top)
        for x, y, w, h, _area in result.boxes:
            x0 = round((x * width / config.IMAGE_WIDTH - left) * sx)
            y0 = round((y * height / config.IMAGE_HEIGHT - top) * sy)
            x1 = round(((x + w) * width / config.IMAGE_WIDTH - left) * sx)
            y1 = round(((y + h) * height / config.IMAGE_HEIGHT - top) * sy)
            cv2.rectangle(rgb, (x0, y0), (x1, y1), color, 2)
        image = QImage(rgb.data, preview_width, preview_height, rgb.strides[0],
                       QImage.Format.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(image.copy()))

        if result.has_reference:
            self.reference_label.setText(self.tr("Reference: OK"))
        elif result.reference_pending:
            self.reference_label.setText(self.tr("Reference: setting..."))
        elif result.status == STATUS_UNCHECKED:
            self.reference_label.setText(self.tr("Reference: capture failed"))
        else:
            self.reference_label.setText(self.tr("Reference: NOT SET"))
        if result.has_reference and (result.status != STATUS_UNCHECKED or result.boxes):
            self.change_label.setText(
                self.tr("Change: {} %").format(round(result.change_ratio * 100, 2)))
        elif result.unchecked_reason == registration.ISSUE_CLIPPING and result.clipping is not None:
            clipped, limit = result.clipping
            self.change_label.setText(self.tr("Clipping: {:.1f}% (limit {:.1f}%)").format(
                clipped * 100, limit * 100))
        elif result.unchecked_reason == registration.ISSUE_OUT_OF_FRAME:
            self.change_label.setText(self.tr("Inspection zone left the frame"))
        elif result.unchecked_reason == registration.ISSUE_ALIGNMENT:
            self.change_label.setText(self.tr("Part alignment failed"))
        else:
            self.change_label.setText(self.tr("Change: -"))

        if result.has_reference:
            self.last_check_label.setText(
                self.tr("Last check: {} ms").format(round(result.elapsed_ms, 1)))
        else:
            self.last_check_label.setText(self.tr("Last check: -"))
        
        self.set_reference_button.setEnabled(True)
        self.reset_roi_button.setEnabled(True)

    def _set_status(self, status: str, *, defect_pending=False):
        texts = {
            STATUS_NORMAL: (self.tr("STATUS: NORMAL"), "#3c3"),
            STATUS_ALARM: (self.tr("STATUS: DEFECT"), "#e33"),
            STATUS_UNCHECKED: (self.tr("STATUS: FRAME NOT CHECKED"), "#ca3"),
            STATUS_NO_REFERENCE: (self.tr("STATUS: NO REFERENCE"), "#aa3"),
            NO_CAMERA: (self.tr("STATUS: NO CAMERA"), "#e33"),
        }
        text, color = texts[status]
        if defect_pending:
            text = self.tr("STATUS: CHECKING DEFECT")
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"font-family: monospace; font-size: 14px; font-weight: bold; color: {color}; background: transparent;"
        )
        if status != self._status:
            self._status = status
            self.status_changed.emit(self.slot_index, status)

    @property
    def status(self) -> str:
        return self._status

    def reset_confirmation(self):
        self.confirmation_reset.emit()
