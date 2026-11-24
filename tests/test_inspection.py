import unittest

import cv2
import numpy as np

import detector
import preprocessing
import registration
from controller import Confirmation, Controller


ZONE = (0.38, 0.38, 0.62, 0.62)


class InspectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(99)
        shape = (256, 256)
        cls.background = rng.integers(45, 115, shape, dtype=np.uint8)
        cls.part = np.zeros(shape, np.uint8)
        cv2.rectangle(cls.part, (55, 45), (198, 210), 255, -1)
        cls.metal = rng.integers(108, 187, shape, dtype=np.uint8)
        cv2.circle(cls.metal, (126, 127), 27, 65, -1)
        for point in ((90, 94), (165, 89), (94, 161), (163, 169)):
            cv2.circle(cls.metal, point, 6, 70, -1)
        cls.light_gradient = np.linspace(0.4, 1, 256)

    def scene(self, dx=0, dy=0, *, light=0, defect=False, outside=False, blur=False, glare=False):
        detail = self.metal.copy()
        if defect:
            cv2.rectangle(detail, (109, 113), (142, 140), 149, -1)
        if outside:
            cv2.rectangle(detail, (104, 68), (148, 87), 61, -1)
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        mask = cv2.warpAffine(self.part, transform, (256, 256), flags=cv2.INTER_NEAREST)
        shifted = cv2.warpAffine(detail, transform, (256, 256))
        frame = np.where(mask != 0, shifted, self.background)
        frame = np.uint8(np.clip(frame + light * self.light_gradient, 0, 255))
        if glare:
            cv2.rectangle(frame, (90, 90), (167, 167), 255, -1)
        if blur:
            frame = cv2.GaussianBlur(frame, (19, 19), 0)
        return frame

    def inspection(self, roi):
        camera = Controller()
        reference = [self.scene(dx, dy) for dx, dy in ((0, 0), (3, -2), (-3, 4), (8, -3), (-10, 7))]
        self.assertTrue(camera.set_reference(reference, roi=roi))
        return camera

    def test_clean_shift_and_light_in_both_modes(self):
        for zone in (None, ZONE):
            camera = self.inspection(zone)
            for dx, dy, light in ((0, 0, 0), (10, -6, 26), (-10, 6, -32), (20, 13, 32), (-24, -22, 18)):
                with self.subTest(zone=zone, dx=dx, dy=dy):
                    result = camera.check_frame(self.scene(dx, dy, light=light))
                    self.assertTrue(result["usable"])
                    self.assertFalse(result["changed"])

    def test_confirmed_defect_follows_part_in_both_modes(self):
        for zone in (None, ZONE):
            with self.subTest(zone=zone):
                camera = self.inspection(zone)
                first = camera.check_frame(self.scene(10, -6, light=26, defect=True))
                second = camera.check_frame(self.scene(-10, 6, light=-32, defect=True))
                self.assertTrue(first["changed"])
                self.assertFalse(first["confirmed"])
                self.assertTrue(second["confirmed"])
                self.assertEqual(second["confirmed_boxes"][0][:4], (99, 119, 34, 28))
                if zone:
                    self.assertEqual(second["offset"], (-10, 6))
                    self.assertAlmostEqual(second["preview_roi"][0], zone[0] - 10 / 256)

    def test_scale_change_with_mixed_reference_preserves_defect_detection(self):
        def zoomed(scale, angle, *, defect=False):
            transform = cv2.getRotationMatrix2D((128, 128), angle, scale)
            return cv2.warpAffine(self.scene(defect=defect), transform, (256, 256),
                                  borderMode=cv2.BORDER_REFLECT_101)

        clean = [self.scene(), zoomed(1.03, 0.4), zoomed(1.045, -0.6)]
        for roi in (None, ZONE):
            with self.subTest(roi=roi):
                camera = Controller()
                self.assertTrue(camera.set_reference([*clean, *clean[1:]], roi=roi))
                for frame in clean:
                    result = camera.check_frame(frame)
                    self.assertTrue(result["usable"])
                    self.assertFalse(result["changed"])
                first = camera.check_frame(zoomed(1.03, 0.4, defect=True))
                second = camera.check_frame(zoomed(1.045, -0.6, defect=True))
                self.assertTrue(first["changed"])
                self.assertFalse(first["confirmed"])
                self.assertTrue(second["confirmed"])

    def test_moving_reflection_preserves_texture_but_plastic_covers_it(self):
        y, x = np.indices((256, 256))

        def reflective_frame(position, *, defect=False):
            ramp = 175 / (1 + np.exp(-(x - 157 + position) / 10))
            reflection = np.exp(-((x - (99 + position)) ** 2 + (y - (175 - 2 * position)) ** 2) / 1150)
            frame = np.uint8(np.clip(self.scene() + ramp + 5 * position * reflection, 0, 255))
            if defect:
                cv2.rectangle(frame, (72, 107), (112, 148), 30, -1)
            return frame

        for roi in (None, (0.24, 0.34, 0.46, 0.62)):
            with self.subTest(roi=roi):
                camera = Controller()
                self.assertTrue(camera.set_reference([reflective_frame(0)] * 5, roi=roi))
                for position in (9, 13):
                    clean = camera.check_frame(reflective_frame(position))
                    self.assertTrue(clean["usable"])
                    self.assertFalse(clean["changed"])
                first = camera.check_frame(reflective_frame(9, defect=True))
                second = camera.check_frame(reflective_frame(13, defect=True))
                self.assertTrue(first["changed"])
                self.assertFalse(first["confirmed"])
                self.assertTrue(second["confirmed"])

    def test_zone_excludes_changes_outside_it(self):
        camera = self.inspection(ZONE)
        result = camera.check_frame(self.scene(10, 7, outside=True))
        self.assertTrue(result["usable"])
        self.assertFalse(result["changed"])
        self.assertEqual(result["changed_pixels"], 0)

    def test_occluder_hiding_zone_and_nearby_anchors_still_confirms(self):
        zone = preprocessing.roi_bounds((256, 256), ZONE)
        for roi in (None, ZONE):
            with self.subTest(roi=roi):
                camera = self.inspection(roi)
                frames = []
                for dx, dy in ((8, -7), (5, -4)):
                    frame = self.scene(dx, dy)
                    cv2.ellipse(frame, (126 + dx, 127 + dy), (80, 120), 14, 0, 360, 198, -1)
                    frames.append(frame)
                if roi:
                    ref = camera.get_reference()
                    alignment = registration.align(ref, preprocessing.preprocess(frames[0]),
                                                   registration.make_anchors(ref), zone)
                    self.assertTrue(alignment.defect_only)
                self.assertFalse(camera.check_frame(frames[0])["confirmed"])
                self.assertTrue(camera.check_frame(frames[1])["confirmed"])
                self.assertFalse(camera.check_frame(self.scene())["changed"])

    def test_distant_matches_alone_cannot_report_normal(self):
        camera = self.inspection(ZONE)
        ref = camera.get_reference()
        zone = preprocessing.roi_bounds(ref.shape, ZONE)
        distant = [anchor for anchor in registration.make_anchors(ref) if anchor.x < 48]
        result = detector.compare(ref, ref, distant, zone)
        self.assertFalse(result["usable"])
        self.assertFalse(result["changed"])

    def test_unchecked_frames_reset_confirmation(self):
        for zone in (None, ZONE):
            for unusable in (self.scene(blur=True), self.scene(glare=True)):
                with self.subTest(zone=zone, frame=unusable.max()):
                    camera = self.inspection(zone)
                    self.assertFalse(camera.check_frame(self.scene(defect=True))["confirmed"])
                    unchecked = camera.check_frame(unusable)
                    self.assertFalse(unchecked["usable"])
                    if unusable.max() == 255:
                        self.assertEqual(unchecked["unchecked_reason"], registration.ISSUE_CLIPPING)
                        fraction, limit = unchecked["clipping"]
                        self.assertGreater(fraction, limit)
                    else:
                        self.assertEqual(unchecked["unchecked_reason"], registration.ISSUE_ALIGNMENT)
                    self.assertFalse(camera.check_frame(self.scene(defect=True))["confirmed"])
                    self.assertTrue(camera.check_frame(self.scene(defect=True))["confirmed"])

    def test_zone_leaving_frame_and_textureless_reference(self):
        camera = self.inspection((0.04, 0.32, 0.27, 0.61))
        # Участок у края следует за всей сценой, включая фон.
        shift = np.float32([[1, 0, -18], [0, 1, 0]])
        moved = cv2.warpAffine(self.scene(), shift, (256, 256))
        result = camera.check_frame(moved)
        self.assertFalse(result["usable"])
        self.assertEqual(result["unchecked_reason"], registration.ISSUE_OUT_OF_FRAME)
        self.assertFalse(camera.set_reference([np.full((256, 256), 120, np.uint8)] * 5))
        self.assertFalse(camera.has_reference())

    def test_unrelated_regions_do_not_confirm(self):
        streak = Confirmation()
        for x in (10, 100, 10):
            box = (x, 10, 15, 15, 225)
            result = {"changed": True, "reference_boxes": [box], "boxes": [box],
                      "components": [np.ones((15, 15), dtype=bool)]}
            self.assertEqual(streak.update(result), [])
        self.assertEqual(streak.update(result), [box])

    def test_overlapping_boxes_need_overlapping_defect_pixels(self):
        streak = Confirmation()
        box = (10, 10, 15, 15, 57)
        mask = np.zeros((15, 15), dtype=bool)
        mask[:2, :] = True
        mask[:, :2] = True
        result = {"changed": True, "reference_boxes": [box], "boxes": [box],
                  "components": [mask]}
        self.assertEqual(streak.update(result), [])
        result["components"] = [mask[::-1, ::-1]]
        self.assertEqual(streak.update(result), [])
        self.assertEqual(streak.update(result), [box])

    def test_repetitive_pattern_cannot_set_reference(self):
        y, x = np.indices((256, 256))
        repeats = np.uint8(np.where((x // 5 + y // 5) % 2, 70, 150))
        camera = Controller()
        self.assertFalse(camera.set_reference([repeats] * 5))


if __name__ == "__main__":
    unittest.main()
