"""Параметры активной камеры OpenCV в единицах её драйвера."""

import json
import logging
import math
from pathlib import Path
from threading import Lock

import cv2

log = logging.getLogger(__name__)
SETTINGS_PATH = Path(__file__).resolve().with_name("opencv_camera_settings.json")
_file_lock = Lock()

# Автоматические режимы применяются перед связанными числовыми параметрами.
PROPERTIES = {
    "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
    "autofocus": "CAP_PROP_AUTOFOCUS",
    "auto_white_balance": "CAP_PROP_AUTO_WB",
    "exposure": "CAP_PROP_EXPOSURE",
    "gain": "CAP_PROP_GAIN",
    "focus": "CAP_PROP_FOCUS",
    "white_balance": "CAP_PROP_WB_TEMPERATURE",
    "brightness": "CAP_PROP_BRIGHTNESS",
    "contrast": "CAP_PROP_CONTRAST",
    "saturation": "CAP_PROP_SATURATION",
    "sharpness": "CAP_PROP_SHARPNESS",
}
MANUAL_PROPERTIES = {
    "auto_exposure": "exposure",
    "autofocus": "focus",
    "auto_white_balance": "white_balance",
}


def _properties_for(backend):
    for name, prop in PROPERTIES.items():
        if name in MANUAL_PROPERTIES and not (
            backend == "DSHOW" or backend == "MSMF" and name != "auto_white_balance"
        ):
            continue
        if backend == "MSMF" and name == "white_balance":
            prop = "CAP_PROP_TEMPERATURE"
        flag = getattr(cv2, prop, None)
        if flag is not None:
            yield name, flag


def _read(cap, backend, name, flag):
    try:
        value = float(cap.get(flag))
    except (cv2.error, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if backend == "DSHOW" and name == "autofocus" and value in (1, 2, 3):
        return int(value) & 1
    if name in MANUAL_PROPERTIES and value not in (0, 1):
        return None
    # -1 в DirectShow означает неизвестное значение, но для выдержки это
    # также корректное значение из логарифмической шкалы драйвера.
    if backend == "DSHOW" and name != "exposure" and value == -1:
        return None
    return value


def read_values(cap, backend):
    return {name: _read(cap, backend, name, flag) for name, flag in _properties_for(backend)}


def apply_values(cap, backend, values):
    values = dict(values)
    for mode, manual in MANUAL_PROPERTIES.items():
        if values.get(mode) == 1:
            values.pop(manual, None)
    accepted, rejected, unverified = {}, {}, []
    for name, flag in _properties_for(backend):
        if name not in values:
            continue
        wanted = values[name]
        if (not isinstance(wanted, (int, float)) or not math.isfinite(wanted) or
                name in MANUAL_PROPERTIES and wanted not in (0, 1)):
            rejected[name] = None
            continue
        try:
            if backend == "MSMF" and name in ("auto_exposure", "autofocus") and wanted == 0:
                # Запись числового значения через MSMF переводит параметр в
                # ручной режим. Так выдержка не получает недопустимое значение 0.
                manual = MANUAL_PROPERTIES[name]
                manual_flag = getattr(cv2, PROPERTIES[manual])
                current = _read(cap, backend, manual, manual_flag)
                success = current is not None and cap.set(manual_flag, current)
            else:
                success = cap.set(flag, wanted)
        except cv2.error:
            success = False
        actual = _read(cap, backend, name, flag)
        if not success or (actual is not None and not math.isclose(wanted, actual, abs_tol=0.001)):
            rejected[name] = actual
        else:
            accepted[name] = wanted if actual is None else actual
            if actual is None:
                unverified.append(name)
    rejected.update({name: None for name in values if name not in accepted and name not in rejected})
    return accepted, rejected, unverified


def _load():
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("Could not read camera settings %s: %s", SETTINGS_PATH, exc)
        return {}


def saved_values(backend, index):
    cameras = _load().get(backend)
    values = cameras.get(str(index)) if isinstance(cameras, dict) else None
    return values if isinstance(values, dict) else {}


def save_values(backend, index, values):
    with _file_lock:
        data = _load()
        if not isinstance(data.get(backend), dict):
            data[backend] = {}
        cameras = data[backend]
        key = str(index)
        old = cameras.get(key)
        cameras[key] = {**(old if isinstance(old, dict) else {}), **values}
        temporary = SETTINGS_PATH.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(SETTINGS_PATH)
        finally:
            temporary.unlink(missing_ok=True)
