"""
IBVAP activity-analysis module

This file is a proposed add-on. It does not modify app.py and does not run by itself.
It consumes detections shaped like:
    {"label": "PERSON", "conf": 0.82, "bbox": (x1, y1, x2, y2)}

The module deliberately produces explainable signals rather than claiming to
identify intent. Operators should calibrate thresholds using local recordings.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


@dataclass
class ActivityConfig:
    # Matching and history
    max_match_distance_px: float = 90.0
    track_ttl_sec: float = 2.0
    history_sec: float = 8.0

    # Geometry. Coordinates are normalized to [0, 1] where possible.
    fence_line: Optional[Tuple[Point, Point]] = None
    protected_polygon: Optional[Sequence[Point]] = None
    crossing_cooldown_sec: float = 4.0

    # Motion rules. These are deliberately conservative starting values.
    abnormal_speed_px_per_sec: float = 260.0
    minimum_motion_samples: int = 4
    direction_change_count: int = 3
    loiter_radius_px: float = 45.0
    loiter_duration_sec: float = 20.0
    loiter_alert_cooldown_sec: float = 30.0

    # Alert quality gates
    min_detection_confidence: float = 0.45
    alert_cooldown_sec: float = 8.0
    persons_only_for_activity: bool = True


@dataclass
class Observation:
    ts: float
    center: Point
    bbox: BBox
    confidence: float
    inside_protected: Optional[bool] = None


@dataclass
class Track:
    track_id: int
    label: str
    observations: deque = field(default_factory=deque)
    last_seen: float = 0.0
    last_alert_by_kind: Dict[str, float] = field(default_factory=dict)
    previous_side_of_fence: Optional[float] = None
    direction_changes: int = 0
    last_direction: Optional[Point] = None


@dataclass
class ActivityAlert:
    kind: str
    level: str
    track_id: int
    label: str
    camera_id: str
    timestamp: float
    reason: str
    evidence: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "level": self.level,
            "track_id": self.track_id,
            "label": self.label,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "evidence": self.evidence,
        }


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


class ActivityEngine:
    """Per-camera, per-session activity state machine."""

    def __init__(self, camera_id: str, config: Optional[ActivityConfig] = None):
        self.camera_id = camera_id
        self.config = config or ActivityConfig()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

    def _eligible(self, detection: Dict[str, object]) -> bool:
        label = str(detection.get("label", ""))
        confidence = float(detection.get("conf", 0.0))
        if confidence < self.config.min_detection_confidence:
            return False
        return (not self.config.persons_only_for_activity) or label == "PERSON"

    def _expire_tracks(self, now: float) -> None:
        expired = [tid for tid, track in self.tracks.items() if now - track.last_seen > self.config.track_ttl_sec]
        for tid in expired:
            del self.tracks[tid]

    def _match_or_create(self, label: str, center: Point, now: float) -> Track:
        candidates = [
            track for track in self.tracks.values()
            if track.label == label and now - track.last_seen <= self.config.track_ttl_sec
        ]
        if candidates:
            closest = min(candidates, key=lambda t: _distance(t.observations[-1].center, center))
            if _distance(closest.observations[-1].center, center) <= self.config.max_match_distance_px:
                return closest
        track = Track(track_id=self.next_track_id, label=label, last_seen=now)
        self.next_track_id += 1
        self.tracks[track.track_id] = track
        return track

    def _can_alert(self, track: Track, kind: str, now: float, cooldown: Optional[float] = None) -> bool:
        wait = self.config.alert_cooldown_sec if cooldown is None else cooldown
        last = track.last_alert_by_kind.get(kind, 0.0)
        if now - last < wait:
            return False
        track.last_alert_by_kind[kind] = now
        return True

    def _make_alert(self, kind: str, level: str, track: Track, now: float, reason: str, evidence: Dict[str, object]) -> ActivityAlert:
        return ActivityAlert(kind, level, track.track_id, track.label, self.camera_id, now, reason, evidence)

    def _motion_metrics(self, track: Track) -> Tuple[float, int, float]:
        samples = list(track.observations)
        if len(samples) < 2:
            return 0.0, 0, 0.0
        speeds = []
        directions = []
        for previous, current in zip(samples[:-1], samples[1:]):
            dt = max(current.ts - previous.ts, 1e-3)
            dx = current.center[0] - previous.center[0]
            dy = current.center[1] - previous.center[1]
            speeds.append(math.hypot(dx, dy) / dt)
            directions.append((dx, dy))
        average_speed = sum(speeds) / len(speeds)
        direction_changes = 0
        for a, b in zip(directions[:-1], directions[1:]):
            if math.hypot(*a) < 2 or math.hypot(*b) < 2:
                continue
            dot = a[0] * b[0] + a[1] * b[1]
            cosine = max(-1.0, min(1.0, dot / (math.hypot(*a) * math.hypot(*b))))
            if math.degrees(math.acos(cosine)) > 100:
                direction_changes += 1
        displacement = _distance(samples[0].center, samples[-1].center)
        return average_speed, direction_changes, displacement

    def update(self, detections: Iterable[Dict[str, object]], timestamp: Optional[float] = None) -> Tuple[List[ActivityAlert], Dict[int, Track]]:
        now = time.time() if timestamp is None else timestamp
        self._expire_tracks(now)
        alerts: List[ActivityAlert] = []

        for detection in detections:
            if not self._eligible(detection):
                continue
            try:
                bbox = tuple(int(v) for v in detection["bbox"])  # type: ignore[index]
                label = str(detection["label"])
                confidence = float(detection["conf"])
            except (KeyError, TypeError, ValueError):
                continue
            center = bbox_center(bbox)  # type: ignore[arg-type]
            track = self._match_or_create(label, center, now)
            inside = None
            if self.config.protected_polygon:
                inside = _point_in_polygon(center, self.config.protected_polygon)
            track.observations.append(Observation(now, center, bbox, confidence, inside))
            track.last_seen = now
            while track.observations and now - track.observations[0].ts > self.config.history_sec:
                track.observations.popleft()

            # A crossing requires two observations on different sides of a line;
            # this avoids declaring a breach from a single noisy box.
            if self.config.fence_line and len(track.observations) >= 2:
                p1, p2 = self.config.fence_line
                side = _cross(p1, p2, center)
                previous = track.previous_side_of_fence
                if previous is not None and side * previous < 0:
                    if self._can_alert(track, "FENCE_CROSSING", now, self.config.crossing_cooldown_sec):
                        alerts.append(self._make_alert(
                            "FENCE_CROSSING", "RED", track, now,
                            "Tracked target crossed the configured virtual-fence line.",
                            {"previous_side": round(previous, 2), "current_side": round(side, 2), "bbox": bbox},
                        ))
                if abs(side) > 1.0:
                    track.previous_side_of_fence = side

            # Polygon intrusion is separate from line crossing and catches a
            # target entering a protected area without crossing the chosen line.
            if inside is True and self._can_alert(track, "ZONE_INTRUSION", now):
                alerts.append(self._make_alert(
                    "ZONE_INTRUSION", "RED", track, now,
                    "Tracked target entered the configured protected polygon.",
                    {"bbox": bbox, "center": tuple(round(v, 1) for v in center)},
                ))

            if len(track.observations) >= self.config.minimum_motion_samples:
                avg_speed, turns, displacement = self._motion_metrics(track)
                if avg_speed >= self.config.abnormal_speed_px_per_sec and self._can_alert(track, "HIGH_SPEED", now):
                    alerts.append(self._make_alert(
                        "HIGH_SPEED", "AMBER", track, now,
                        "Sustained image-plane speed exceeded the configured threshold; review video context.",
                        {"avg_speed_px_per_sec": round(avg_speed, 1), "displacement_px": round(displacement, 1)},
                    ))
                if turns >= self.config.direction_change_count and self._can_alert(track, "ERRATIC_MOTION", now):
                    alerts.append(self._make_alert(
                        "ERRATIC_MOTION", "AMBER", track, now,
                        "Repeated direction reversals were observed; this is an uncertainty signal, not intent classification.",
                        {"direction_changes": turns, "displacement_px": round(displacement, 1)},
                    ))

                oldest = track.observations[0]
                if now - oldest.ts >= self.config.loiter_duration_sec and displacement <= self.config.loiter_radius_px:
                    if self._can_alert(track, "LOITERING", now, self.config.loiter_alert_cooldown_sec):
                        alerts.append(self._make_alert(
                            "LOITERING", "AMBER", track, now,
                            "Target remained within a small image-plane radius for the configured duration.",
                            {"duration_sec": round(now - oldest.ts, 1), "displacement_px": round(displacement, 1)},
                        ))

        return alerts, dict(self.tracks)


def detections_from_current_app(detections: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Compatibility helper: the current app must add bbox to each detection first."""
    return [d for d in detections if "bbox" in d]


__all__ = ["ActivityAlert", "ActivityConfig", "ActivityEngine", "Track", "detections_from_current_app"]

def draw_activity_overlay(frame, engine: ActivityEngine) -> object:
    """Optional OpenCV overlay helper; import cv2 only when this function is used."""
    import cv2
    for track in engine.tracks.values():
        if not track.observations:
            continue
        obs = track.observations[-1]
        x1, y1, x2, y2 = obs.bbox
        color = (0, 0, 255) if obs.inside_protected else (255, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track.track_id} {track.label}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        trail = list(track.observations)[-20:]
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(frame, tuple(map(int, a.center)), tuple(map(int, b.center)), color, 2)
    if engine.config.fence_line:
        a, b = engine.config.fence_line
        cv2.line(frame, tuple(map(int, a)), tuple(map(int, b)), (0, 0, 255), 2)
    return frame
