from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

import config

ISSUE_ALIGNMENT = "alignment"
ISSUE_OUT_OF_FRAME = "out_of_frame"
ISSUE_CLIPPING = "clipping"


@dataclass
class Anchor:
    x: int
    y: int
    patch: np.ndarray
    sharpness: float


@dataclass
class Match:
    anchor: Anchor
    dx: float
    dy: float
    brightness: float


@dataclass
class Alignment:
    image: np.ndarray
    valid: np.ndarray
    brightness: np.ndarray
    layers: list
    usable: bool
    defect_only: bool = False
    scaled: bool = False
    issue: str | None = ISSUE_ALIGNMENT
    clipping: tuple[float, float] | None = None


def gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def make_anchors(image):
    image = gray(image)
    height, width = image.shape
    radius = config.REGISTRATION_PATCH_SIZE // 2
    step = config.REGISTRATION_GRID_STEP
    corners = cv2.cornerMinEigenVal(image, 5)
    anchors = []
    for top in range(radius, height - radius, step):
        for left in range(radius, width - radius, step):
            cell = corners[top:min(top + step, height - radius),
                           left:min(left + step, width - radius)]
            _, strength, _, (cx, cy) = cv2.minMaxLoc(cell)
            if strength < 0.00012:
                continue
            x, y = left + cx, top + cy
            patch = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
            sharpness = float(cv2.Laplacian(patch, cv2.CV_32F).std())
            if sharpness >= 5 and patch.std() >= 7:
                anchors.append(Anchor(x, y, patch.copy(), sharpness))
    return anchors


def _peak_offset(before, peak, after):
    curvature = before - 2.0 * peak + after
    return float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5)) if curvature < 0 else 0.0


def _match(image, anchor):
    limit = config.REGISTRATION_MAX_SHIFT
    size = anchor.patch.shape[0]
    radius = size // 2
    left = max(0, anchor.x - radius - limit)
    top = max(0, anchor.y - radius - limit)
    right = min(image.shape[1], anchor.x + radius + limit + 1)
    bottom = min(image.shape[0], anchor.y + radius + limit + 1)
    search = image[top:bottom, left:right]
    scores = cv2.matchTemplate(search, anchor.patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(scores)
    if score < config.REGISTRATION_MIN_SCORE:
        return None
    sub_x = _peak_offset(scores[y, x - 1], score, scores[y, x + 1]) if 0 < x < scores.shape[1] - 1 else 0
    sub_y = _peak_offset(scores[y - 1, x], score, scores[y + 1, x]) if 0 < y < scores.shape[0] - 1 else 0
    scores[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = -1
    if score - cv2.minMaxLoc(scores)[1] < config.REGISTRATION_PEAK_MARGIN:
        return None
    patch = search[y:y + size, x:x + size]
    if cv2.Laplacian(patch, cv2.CV_32F).std() < 0.45 * anchor.sharpness:
        return None
    brightness = np.median(patch.astype(np.int16) - anchor.patch)
    return Match(anchor, left + x + radius - anchor.x + sub_x,
                 top + y + radius - anchor.y + sub_y, float(brightness))


def _groups(matches):
    """Собирает совпавшие ориентиры с одинаковым переносом."""
    if not matches:
        return []
    shifts = np.array([(m.dx, m.dy) for m in matches])
    agreement = np.max(np.abs(shifts[:, None] - shifts), axis=2) <= 1.4
    groups = []
    remaining = np.ones(len(matches), dtype=bool)
    while remaining.any():
        votes = (agreement & remaining).sum(axis=1)
        votes[~remaining] = 0
        best = int(votes.argmax())
        if votes[best] < 2:
            break
        members = agreement[best] & remaining
        offset = np.median(shifts[members], axis=0)
        groups.append((offset, [matches[i] for i in np.flatnonzero(members)]))
        remaining[members] = False
    return groups


def _similarity(matches, shape):
    if len(matches) < 8:
        return None
    source = np.float32([(m.anchor.x, m.anchor.y) for m in matches])
    target = source + np.float32([(m.dx, m.dy) for m in matches])
    transform, inliers = cv2.estimateAffinePartial2D(source, target, ransacReprojThreshold=1.1)
    if transform is None:
        return None
    inliers = inliers.ravel().astype(bool)
    if inliers.sum() < max(8, 0.68 * len(matches)):
        return None
    height, width = shape
    if np.any(np.ptp(source[inliers], axis=0) < (0.35 * width, 0.35 * height)):
        return None
    scale = hypot(transform[0, 0], transform[0, 1])
    if not 0.90 <= scale <= 1.10 or abs(transform[0, 1]) > 0.10:
        return None
    corners = np.float32([(0, 0), (width, 0), (0, height), (width, height)])
    displacement = cv2.transform(corners[None], transform)[0] - corners
    if (np.abs(displacement).max() > config.REGISTRATION_MAX_SHIFT + 1 or
            np.ptp(displacement, axis=0).max() < 1.5):
        return None
    shifts = target[inliers] - source[inliers]
    if (np.abs(shifts - np.median(shifts, axis=0)).max(axis=1) <= 1.1).mean() > 0.80:
        return None
    return transform, [match for match, trusted in zip(matches, inliers) if trusted]


def transform_bounds(bounds, transform):
    left, top, right, bottom = bounds
    corners = np.float32([[(left, top), (right, top), (left, bottom), (right, bottom)]])
    warped = cv2.transform(corners, transform)[0]
    return (*warped.min(axis=0), *warped.max(axis=0))


def _translation(offset):
    dx, dy = (round(v) if abs(v - round(v)) < 0.16 else v for v in offset)
    return np.float32([[1, 0, dx], [0, 1, dy]])


def _distance_to_zone(anchor, zone):
    left, top, right, bottom = zone
    return hypot(max(left - anchor.x, anchor.x - right, 0),
                 max(top - anchor.y, anchor.y - bottom, 0))


def _zone_group(groups, zone):
    radius = config.REGISTRATION_PATCH_SIZE // 2
    reach = 2 * config.REGISTRATION_GRID_STEP
    left, top, right, bottom = zone
    candidates = []
    for offset, near in groups:
        outside = [m for m in near if (
            m.anchor.x + radius < left or m.anchor.x - radius >= right or
            m.anchor.y + radius < top or m.anchor.y - radius >= bottom
        )]
        if len(near) >= 2:
            candidates.append((offset, near, outside))
    exterior = [item for item in candidates if len(item[2]) >= 2]
    if exterior:
        # Близкие ориентиры важнее дальнего фона. При равных свидетельствах
        # внутри зоны используем только согласованный внешний перенос.
        ranked = []
        for offset, near, outside in exterior:
            support = sum(1.0 - _distance_to_zone(m.anchor, zone) / (reach + 1)
                          for m in outside)
            inside = len(near) - len(outside)
            ranked.append((support + min(inside, 2) * 0.45, offset, outside))
    else:
        ranked = [(float(len(near)), offset, near) for offset, near, _ in candidates]
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked or (len(ranked) > 1 and ranked[1][0] >= ranked[0][0] * 0.80):
        return []
    _, offset, members = ranked[0]
    return [(offset, members)]


def _distance_map(shape, members):
    seeds = np.ones(shape, dtype=np.uint8)
    for match in members:
        seeds[match.anchor.y, match.anchor.x] = 0
    return cv2.distanceTransform(seeds, cv2.DIST_L2, cv2.DIST_MASK_3)


def _brightness_map(shape, members):
    values = np.zeros(shape, dtype=np.float32)
    weights = np.zeros_like(values)
    for match in members:
        y, x = match.anchor.y, match.anchor.x
        values[y, x] = match.brightness
        weights[y, x] = 1
    sigma = config.REGISTRATION_GRID_STEP
    values = cv2.GaussianBlur(values, (0, 0), sigma)
    weights = cv2.GaussianBlur(weights, (0, 0), sigma)
    fallback = np.full_like(values, np.median([m.brightness for m in members]))
    return np.divide(values, weights, out=fallback, where=weights > 1e-6)


def align(reference, current, anchors, zone=None):
    """Переносит кадр в координаты эталона и отмечает проверяемые пиксели."""
    shape = reference.shape[:2]
    valid = np.zeros(shape, dtype=bool)
    aligned = np.zeros_like(reference)
    brightness = np.zeros(shape, dtype=np.float32)
    layers = []
    result = Alignment(aligned, valid, brightness, layers, False)
    if reference.shape != current.shape:
        return result

    current_gray = gray(current)
    matches = [match for anchor in anchors if (match := _match(current_gray, anchor))]
    fallback = False
    similarity = _similarity(matches, shape)
    if similarity is not None:
        result.scaled = True
        models = [similarity]
    elif zone:
        reach = 2 * config.REGISTRATION_GRID_STEP
        nearby = [m for m in matches if _distance_to_zone(m.anchor, zone) <= reach]
        groups = _zone_group(_groups(nearby), zone)
        if not groups:
            # Препятствие может закрыть и зону, и её ориентиры. Дальние
            # совпадения подтверждают видимость сцены, но не перенос детали.
            groups = [(np.zeros(2), members) for _, members in _groups(matches)[:1]
                      if len(members) >= 3]
            fallback = bool(groups)
        models = [(_translation(offset), members) for offset, members in groups]
    else:
        models = [(_translation(offset), members) for offset, members in _groups(matches)
                  if len(members) >= 3][:3]
    if not models:
        return result

    height, width = shape
    if zone:
        left, top, right, bottom = transform_bounds(zone, models[0][0])
        if left < -0.25 or top < -0.25 or right > width + 0.25 or bottom > height + 0.25:
            result.issue = ISSUE_OUT_OF_FRAME
            return result
    elif not result.scaled:
        distance_maps = np.stack([_distance_map(shape, members) for _, members in models])
        nearest = np.argmin(distance_maps, axis=0)
        closest = distance_maps.min(axis=0)

    support_radius = 4 * config.REGISTRATION_GRID_STEP
    base_gray = gray(reference)
    for index, (transform, members) in enumerate(models):
        warped = cv2.warpAffine(current, transform, (width, height),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        in_frame = cv2.warpAffine(np.ones(shape, np.uint8), transform, (width, height),
                                  flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
        adjustment = _brightness_map(shape, members)
        if zone or result.scaled:
            supported = in_frame
        else:
            owned = (nearest == index) & (closest <= support_radius)
            supported = owned.copy()
            for other, (other_transform, _) in enumerate(models):
                if index != other:
                    gap = config.REGISTRATION_PATCH_SIZE // 2 + np.linalg.norm(
                        transform[:, 2] - other_transform[:, 2])
                    supported &= distance_maps[other] - closest > gap
            # Возле границы слоёв учитываем только пиксели, чьё соответствие
            # подтверждают и яркость, и большинство соседних пикселей.
            delta = np.abs(gray(warped).astype(np.float32) - adjustment - base_gray)
            close = (delta <= max(5, config.PIXEL_THRESHOLD * 0.55)) & in_frame
            consistent = cv2.blur(close.astype(np.float32), (7, 7)) >= 0.75
            supported = (supported | (owned & close & consistent)) & in_frame
        valid |= supported
        aligned[supported] = warped[supported]
        # После клиппинга точная яркость неизвестна. Поправка по матовым
        # участкам не должна превращать белую область в серую.
        highlight_weight = np.clip((250 - gray(warped).astype(np.float32)) / 35, 0, 1)
        brightness[supported] = (adjustment * highlight_weight)[supported]
        layers.append((transform, supported))

    region = np.s_[zone[1]:zone[3], zone[0]:zone[2]] if zone else np.s_[:, :]
    inspected = valid[region]
    coverage = inspected.mean()
    if coverage < (1.0 if zone else 0.35):
        if zone:
            result.issue = ISSUE_OUT_OF_FRAME
        return result
    base = base_gray[region]
    observed = gray(aligned)[region]
    newly_clipped = (((observed >= 250) & (base < 245)) |
                     ((observed <= 3) & (base > 8))) & inspected
    clipping_limit = (config.REGISTRATION_MAX_NEW_CLIPPING if zone else
                      config.REGISTRATION_MAX_FULL_FRAME_CLIPPING)
    clipped_fraction = float(newly_clipped.sum() / inspected.sum())
    if clipped_fraction > clipping_limit:
        result.issue = ISSUE_CLIPPING
        result.clipping = (clipped_fraction, clipping_limit)
        return result
    result.defect_only = fallback or (not zone and coverage < config.REGISTRATION_MIN_COVERAGE)
    result.usable = not result.defect_only
    if result.usable:
        result.issue = None
    return result
