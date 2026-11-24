import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import cv2

import camera
import opencv_controls as controls


class Driver:
    def __init__(self, backend="DSHOW"):
        self.backend = backend
        self.values = {
            cv2.CAP_PROP_EXPOSURE: -6,
            cv2.CAP_PROP_AUTOFOCUS: 2,
            cv2.CAP_PROP_BRIGHTNESS: 120,
        }
        self.writes = []
        self.released = False

    def getBackendName(self):
        return self.backend

    def get(self, prop):
        return self.values.get(prop, -1)

    def set(self, prop, value):
        self.writes.append((prop, value))
        if prop == cv2.CAP_PROP_GAIN:
            return False
        if prop == cv2.CAP_PROP_EXPOSURE and self.backend == "MSMF":
            self.values[cv2.CAP_PROP_AUTO_EXPOSURE] = 0
        if prop != cv2.CAP_PROP_AUTO_EXPOSURE:
            self.values[prop] = value
        return True

    def isOpened(self):
        return True

    def release(self):
        self.released = True


class OpenCVControlsTest(unittest.TestCase):
    def test_dshow_applies_manual_mode_before_exposure(self):
        driver = Driver()
        accepted, rejected, unverified = controls.apply_values(
            driver, "DSHOW", {"gain": 4, "exposure": -8, "auto_exposure": 0})
        self.assertEqual(driver.writes, [
            (cv2.CAP_PROP_AUTO_EXPOSURE, 0),
            (cv2.CAP_PROP_EXPOSURE, -8),
            (cv2.CAP_PROP_GAIN, 4),
        ])
        self.assertEqual(accepted, {"auto_exposure": 0, "exposure": -8})
        self.assertEqual(rejected, {"gain": None})
        self.assertEqual(unverified, ["auto_exposure"])
        self.assertEqual(controls.read_values(driver, "DSHOW")["autofocus"], 0)

    def test_msmf_can_disable_auto_exposure_without_writing_zero_exposure(self):
        driver = Driver("MSMF")
        driver.values[cv2.CAP_PROP_AUTO_EXPOSURE] = 1
        accepted, rejected, _ = controls.apply_values(driver, "MSMF", {"auto_exposure": 0})
        self.assertEqual(driver.writes, [(cv2.CAP_PROP_EXPOSURE, -6)])
        self.assertEqual(accepted, {"auto_exposure": 0})
        self.assertEqual(rejected, {})

    def test_auto_mode_ignores_saved_manual_value(self):
        driver = Driver()
        controls.apply_values(driver, "DSHOW", {"exposure": -8, "auto_exposure": 1})
        self.assertEqual(driver.writes, [(cv2.CAP_PROP_AUTO_EXPOSURE, 1)])

    def test_ignoring_a_write_is_reported(self):
        driver = Driver()
        with patch.object(driver, "set", return_value=True):
            accepted, rejected, unverified = controls.apply_values(driver, "DSHOW", {"exposure": -8})
        self.assertFalse(accepted)
        self.assertEqual(rejected, {"exposure": -6})
        self.assertFalse(unverified)

    def test_live_settings_are_saved_and_restored_for_the_same_camera(self):
        path = Path.cwd() / f".test-camera-settings-{uuid4().hex}.json"
        self.addCleanup(path.unlink, missing_ok=True)
        with patch.object(
            controls, "SETTINGS_PATH", path
        ), patch("camera.sys.platform", "win32"), patch("camera.cv2.VideoCapture") as open_capture:
            first, second = Driver(), Driver()
            open_capture.side_effect = [first, second]
            active = camera.Camera()
            self.assertTrue(active.open(2))
            rejected, unverified, error = active.apply_opencv_controls(2, {"exposure": -8, "gain": 4})
            self.assertEqual(rejected, {"gain": None})
            self.assertFalse(unverified)
            self.assertIsNone(error)
            self.assertEqual(controls.saved_values("DSHOW", 2), {"exposure": -8})
            self.assertFalse(controls.saved_values("DSHOW", 0))
            self.assertFalse(controls.saved_values("MSMF", 2))
            active.close()
            self.assertTrue(first.released)
            self.assertTrue(active.open(2))
            self.assertEqual(active.read_opencv_controls()[1]["exposure"], -8)
            active.close()
        self.assertEqual(open_capture.call_args.args, (2, cv2.CAP_DSHOW))

    def test_default_backends_follow_platform(self):
        for platform, flags in (
            ("win32", [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]),
            ("linux", [cv2.CAP_V4L2, cv2.CAP_ANY]),
            ("darwin", [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]),
        ):
            with self.subTest(platform=platform), patch("camera.sys.platform", platform):
                self.assertEqual(camera._opencv_backend_flags(), flags)

    def test_linux_v4l2_settings_take_precedence_when_available(self):
        driver = Driver("V4L2")
        with patch("camera.sys.platform", "linux"), patch("camera.v4l2.available", return_value=True), patch(
            "camera.v4l2.apply_for_index", return_value=True
        ) as apply_v4l2, patch("camera.cv2.VideoCapture", return_value=driver), patch.object(
            controls, "saved_values", return_value={"exposure": -8}
        ) as read_opencv:
            active = camera.Camera()
            self.assertTrue(active.open(0))
            self.assertEqual(active.read_opencv_controls()[1]["exposure"], -6)
            active.close()
        apply_v4l2.assert_called_once_with(0)
        read_opencv.assert_not_called()
