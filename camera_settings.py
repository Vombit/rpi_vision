"""Диалог правки параметров камеры (V4L2) прямо из программы.

Показывает РЕАЛЬНЫЕ контролы устройства (`v4l2-ctl --list-ctrls`), их
диапазоны и текущие значения. Значения из config.V4L2_CONTROLS
показываются рядом как подсказка и помечаются ✓ / ≠.

Apply сначала применяет, затем ПЕРЕЧИТЫВАЕТ камеру и показывает,
что реально встало, а что нет. Диалог не закрывается, пока есть
неприменённые значения — чтобы не было иллюзии «применилось».
"""

import logging

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import config
from opencv_controls import MANUAL_PROPERTIES
import v4l2

log = logging.getLogger(__name__)

TYPE_LABELS = {
    "int": "int", "integer": "int", "boolean": "bool", "bitmask": "int",
    "int64": "int", "int32": "int", "ctrl_class": "class", "string": "text",
}


class CameraSettingsDialog(QDialog):
    """Параметры одной камеры. Apply: значения применены и проверены."""

    applying = Signal()

    def __init__(self, camera_index, device, applied_values, parent=None):
        super().__init__(parent)
        self.camera_index = int(camera_index)
        self.device = device
        self._editors = {}
        self._config_values = dict(applied_values or {})

        self.setWindowTitle(self.tr("Camera settings: {}").format(device))
        # Тач-режим: диалог крупный, кнопки — палец.
        self.setMinimumSize(720, 620)
        self._apply_touch_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel(self.tr("Device: {}").format(device))
        header.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(header)

        controls = v4l2.list_controls(device)
        if not controls:
            QMessageBox.warning(
                self, self.tr("Camera settings"),
                self.tr("Could not read camera controls: {}.\n"
                        "Check that v4l2-ctl is installed and the camera is free.").format(device))
            controls = {}
        else:
            layout.addWidget(self._build_form(controls))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.apply_button = QPushButton(self.tr("Apply"))
        self.apply_button.setEnabled(bool(controls))
        self.apply_button.clicked.connect(self._on_apply)
        buttons.addButton(self.apply_button, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ---------- UI ----------

    def _apply_touch_style(self):
        self.setStyleSheet("""
            * { font-size: 22px; }
            QPushButton {
                min-height: 72px;
                min-width: 140px;
                padding: 12px 20px;
                border-radius: 12px;
                border: 2px solid #888;
                background: #ffffff;
            }
            QPushButton:pressed { background: #d8d8d8; }
            QPushButton:disabled { color: #999; border-color: #ccc; }
            QSpinBox, QComboBox {
                min-height: 56px;
                padding: 6px 10px;
                font-size: 22px;
                border-radius: 8px;
                border: 2px solid #888;
            }
        """)

    def _build_form(self, controls) -> QWidget:
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        for name, info in controls.items():
            editor = self._build_editor(info)
            self._editors[name] = editor
            form.addRow(self._build_label(name, info), editor)

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(form_host)
        area.setMinimumSize(620, 380)
        return area

    def _build_label(self, name, info) -> QLabel:
        suffix = ""
        if info["choices"]:
            suffix = f" [{TYPE_LABELS.get(info['type'], info['type'])}]"
        elif not info["writable"]:
            suffix = f" ({self.tr('read-only')})"

        hint = ""
        want = self._config_values.get(name)
        if want is not None:
            actual = info["value"]
            if want == actual:
                hint = f"\n{self.tr('config')}: {want} ✓"
            else:
                hint = f"\n{self.tr('config')}: {want} ≠ {self.tr('actual')} {actual}"

        label = QLabel(f"{name}{suffix}{hint}")
        label.setTextFormat(Qt.TextFormat.PlainText)
        base = "font-size: 20px;"
        if hint.endswith("✓"):
            label.setStyleSheet(base + " color: #202020;")
        elif "≠" in hint:
            label.setStyleSheet(base + " color: #b03030;")
        else:
            label.setStyleSheet(base)
        return label

    def _build_editor(self, info):
        # ВАЖНО: всегда реальное значение с камеры, не из конфига.
        value = info["value"]

        if info["choices"]:
            combo = QComboBox()
            for choice_value, choice_label in info["choices"]:
                combo.addItem(f"{choice_value}: {choice_label}", userData=choice_value)
            pos = combo.findData(int(value))
            combo.setCurrentIndex(max(pos, 0))
            combo.setEnabled(info["writable"])
            return combo

        spin = QSpinBox()
        spin.setRange(int(info["min"]), int(info["max"]))
        spin.setSingleStep(max(int(info["step"]), 1))
        try:
            spin.setValue(int(value))
        except (TypeError, ValueError):
            spin.setValue(int(info["min"]))
        spin.setEnabled(info["writable"])
        return spin

    # ---------- data ----------

    def values(self) -> dict:
        out = {}
        for name, editor in self._editors.items():
            if isinstance(editor, QComboBox):
                data = editor.currentData()
                if data is not None:
                    out[name] = int(data)
            else:
                out[name] = int(editor.value())
        return out

    def _refresh_editors(self, controls):
        for name, info in controls.items():
            editor = self._editors.get(name)
            if editor is None:
                continue
            if isinstance(editor, QComboBox):
                pos = editor.findData(int(info["value"]))
                if pos >= 0:
                    editor.setCurrentIndex(pos)
            else:
                try:
                    editor.blockSignals(True)
                    editor.setValue(int(info["value"]))
                finally:
                    editor.blockSignals(False)

    # ---------- apply ----------

    def _on_apply(self):
        requested = self.values()
        if not requested:
            self.accept()
            return

        try:
            self.applying.emit()
            v4l2.apply_values(self.camera_index, requested)
        except Exception as exc:  # noqa: BLE001 — диалог не должен падать
            log.warning("v4l2 apply failed: %s", exc)
            QMessageBox.warning(
                self, self.tr("Camera settings"),
                self.tr("Failed to apply settings: {}").format(exc))
            return

        # Перечитываем РЕАЛЬНОЕ состояние камеры
        controls_after = v4l2.list_controls(self.device)
        if not controls_after:
            QMessageBox.warning(
                self, self.tr("Camera settings"),
                self.tr("Applied, but could not re-read camera to verify."))
            self.accept()
            return

        actual = {name: info["value"] for name, info in controls_after.items()}
        not_supported = []
        stuck = []
        for name, wanted in requested.items():
            if name not in actual:
                not_supported.append(name)
                continue
            
            actual_val = actual[name]
            
            # Читаем шаг контрола (например, 5 или 10). Если шаг не указан, берем 1.
            step = max(1, int(controls_after[name].get("step", 1)))
            
            # Если разница БОЛЬШЕ шага — только тогда это реальная ошибка.
            # Если разница 5 (5005 -> 5000), а шаг 5 — всё ок, молчим.
            if abs(actual_val - wanted) > step:
                stuck.append(f"{name}: {self.tr('asked')} {wanted}, "
                             f"{self.tr('has')} {actual_val}")

        if not_supported or stuck:
            parts = []
            if not_supported:
                parts.append(
                    self.tr("Not supported by camera:") + "\n  " + "\n  ".join(not_supported))
            if stuck:
                parts.append(
                    self.tr("Requested but not applied:") + "\n  " + "\n  ".join(stuck))
            QMessageBox.warning(self, self.tr("Camera settings"), "\n\n".join(parts))

            # Показать реальные значения, оставить диалог открытым
            self._refresh_editors(controls_after)
            return

        log.info("Camera settings applied to %s: %d control(s)", self.device, len(requested))
        self.accept()


class OpenCVCameraSettingsDialog(QDialog):
    applying = Signal()

    def __init__(self, capture, parent=None):
        super().__init__(parent)
        self._capture = capture
        self._token = object()
        self._baseline = {}
        self._editors = {}
        self._names = {
            "auto_exposure": self.tr("Auto exposure"),
            "autofocus": self.tr("Autofocus"),
            "auto_white_balance": self.tr("Auto white balance"),
            "exposure": self.tr("Exposure"),
            "gain": self.tr("Gain"),
            "focus": self.tr("Focus"),
            "white_balance": self.tr("White balance"),
            "brightness": self.tr("Brightness"),
            "contrast": self.tr("Contrast"),
            "saturation": self.tr("Saturation"),
            "sharpness": self.tr("Sharpness"),
        }
        self.setWindowTitle(self.tr("Camera settings"))
        self.setMinimumSize(560, 460)
        layout = QVBoxLayout(self)
        self._message = QLabel(self.tr("Reading camera controls…"))
        self._message.setWordWrap(True)
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._message)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._form_host = QWidget()
        self._form = QFormLayout(self._form_host)
        scroll.setWidget(self._form_host)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply |
                                   QDialogButtonBox.StandardButton.Close)
        self._apply = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self._apply.setText(self.tr("Apply"))
        buttons.button(QDialogButtonBox.StandardButton.Close).setText(self.tr("Close"))
        self._apply.setEnabled(False)
        self._apply.clicked.connect(self._on_apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        capture.settings_read.connect(self._on_read)
        capture.settings_applied.connect(self._on_applied)
        capture.finished.connect(self._unavailable)
        capture.request_settings(self._token)

    @Slot()
    def _unavailable(self):
        self._message.setText(self.tr("Controls are unavailable for this camera backend."))
        self._apply.setEnabled(False)
        self._form_host.setEnabled(False)

    @Slot(object, str, object)
    def _on_read(self, token, backend, values):
        if token is not self._token:
            return
        if not values:
            self._unavailable()
            return
        self._message.setText(self.tr("{} · Values use camera driver units.").format(backend))
        for name, value in values.items():
            if name in MANUAL_PROPERTIES:
                editor = QComboBox()
                editor.addItem(self.tr("Unknown"), None)
                editor.addItem(self.tr("Manual"), 0)
                editor.addItem(self.tr("Automatic"), 1)
                editor.currentIndexChanged.connect(self._sync_manual_controls)
            else:
                editor = QDoubleSpinBox()
                editor.setRange(-1_000_000, 1_000_000)
                editor.setDecimals(3)
                editor.setSpecialValueText(self.tr("Unknown"))
            self._editors[name] = editor
            self._form.addRow(self._names[name], editor)
            self._show_value(name, value)
        self._sync_manual_controls()
        self._apply.setEnabled(True)

    def _sync_manual_controls(self):
        for mode, manual in MANUAL_PROPERTIES.items():
            if mode in self._editors and manual in self._editors:
                self._editors[manual].setEnabled(self._value(self._editors[mode]) != 1)

    def _show_value(self, name, value):
        editor = self._editors[name]
        if isinstance(editor, QComboBox):
            editor.setCurrentIndex(max(0, editor.findData(value)))
        else:
            editor.setValue(editor.minimum() if value is None else value)
        self._baseline[name] = self._value(editor)

    @staticmethod
    def _value(editor):
        if isinstance(editor, QComboBox):
            return editor.currentData()
        number = editor.value()
        return None if number == editor.minimum() else number

    @Slot()
    def _on_apply(self):
        requested = {}
        for name, editor in self._editors.items():
            value = self._value(editor)
            if editor.isEnabled() and value is not None and value != self._baseline[name]:
                requested[name] = value
        if not requested:
            return
        self._apply.setEnabled(False)
        self._form_host.setEnabled(False)
        self._message.setText(self.tr("Applying camera controls…"))
        self.applying.emit()
        self._capture.request_settings_apply(self._token, requested)

    @Slot(object, object, object, object, object)
    def _on_applied(self, token, actual, rejected, unverified, save_error):
        if token is not self._token:
            return
        for name, value in actual.items():
            self._show_value(name, value)
        messages = []
        if rejected:
            names = ", ".join(self._names[name] for name in rejected)
            messages.append(self.tr("Driver rejected or changed: {}").format(names))
        if unverified:
            names = ", ".join(self._names[name] for name in unverified)
            messages.append(self.tr("Driver accepted without readback: {}").format(names))
        if save_error:
            messages.append(self.tr("Could not save settings: {}").format(save_error))
        if not messages:
            messages.append(self.tr("Settings applied and saved."))
        self._message.setText("\n".join(messages))
        self._form_host.setEnabled(True)
        self._sync_manual_controls()
        self._apply.setEnabled(True)
