"""Dependency-light helpers for the optional SAM 2 tracking prototype.

The production tracker does not import SAM 2.  Only ``load_predictor`` performs
the optional import, so BHASKARA remains runnable without the prototype model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np


OBJECT_LABELS = (
    "wired earphones", "wireless earbuds", "remote control", "mobile phone",
    "metal ruler", "medicine box", "usb cable", "id card", "glasses",
    "charger", "keys", "wallet", "bottle", "book", "watch", "scissors",
    "keyboard", "mouse", "clothes", "fan", "pen", "cup", "laptop",
    "cabinet", "window", "door", "desk", "table", "chair", "shelf",
    "sofa", "bed",
)


SEMANTIC_PROMOTION_ALTERNATIVES = {
    "metal ruler": ("metal ruler", "lanyard", "ribbon", "strap"),
    "id card": (
        "id card", "hanging air freshener", "product tag", "medal",
        "floor tile", "tile grout pattern",
    ),
    "usb cable": ("usb cable", "wired earphones", "strap", "cord"),
    "wired earphones": ("wired earphones", "usb cable", "lanyard", "cord"),
    "charger": ("charger", "power socket", "power adapter", "wall fixture"),
    "keys": ("keys", "keychain", "metal ornament", "small tool"),
}


@dataclass(frozen=True)
class MaskObservation:
    track_id: int
    frame_index: int
    box: tuple[int, int, int, int]
    area: int
    visible: bool


def mask_to_box(mask: np.ndarray, minimum_area: int = 16) -> tuple[int, int, int, int] | None:
    """Return an exclusive-maximum XYXY box, or None for an unusable mask."""
    binary = np.asarray(mask).squeeze().astype(bool)
    if binary.ndim != 2:
        raise ValueError("SAM 2 mask must reduce to a two-dimensional array")

    ys, xs = np.nonzero(binary)
    if xs.size < minimum_area:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_observation(
    track_id: int,
    frame_index: int,
    mask: np.ndarray,
    minimum_area: int = 16,
) -> MaskObservation:
    binary = np.asarray(mask).squeeze().astype(bool)
    box = mask_to_box(binary, minimum_area=minimum_area)
    return MaskObservation(
        track_id=track_id,
        frame_index=frame_index,
        box=box or (0, 0, 0, 0),
        area=int(binary.sum()),
        visible=box is not None,
    )


def load_predictor(model_id: str, device: str) -> Any:
    """Load SAM 2 lazily and provide an actionable error when absent."""
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM 2 is not installed. Install it in an isolated environment "
            "before running experiments/sam2_video_prototype.py."
        ) from exc

    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    return predictor


def sorted_frame_paths(frame_directory: str | Path) -> list[Path]:
    directory = Path(frame_directory)
    return sorted(directory.glob("*.jpg"), key=lambda path: int(path.stem))


def label_candidates(raw_label: str) -> list[str]:
    """Extract all known labels from a Grounding DINO phrase."""
    text = raw_label.lower().strip()
    return [label for label in OBJECT_LABELS if label in text]


def semantic_promotion_candidates(label: str) -> tuple[str, ...]:
    """Return look-alikes requiring verification before track promotion."""
    return SEMANTIC_PROMOTION_ALTERNATIVES.get(label, ())


def semantic_promotion_approved(
    detector_label: str,
    result: dict,
    minimum_score: float = 0.55,
    minimum_margin: float = 0.15,
) -> bool:
    return (
        result.get("best_label") == detector_label
        and float(result.get("best_score", 0.0)) >= minimum_score
        and float(result.get("margin", 0.0)) >= minimum_margin
    )


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def box_coverages(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Return intersection divided by each box's area."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return (
        intersection / first_area if first_area else 0.0,
        intersection / second_area if second_area else 0.0,
    )


def duplicates_existing_track(
    detection: dict,
    track_boxes: dict[int, tuple[int, int, int, int]],
    track_labels: dict[int, str],
    minimum_iou: float = 0.20,
    minimum_containment: float = 0.65,
) -> bool:
    """Block partial same-label boxes from becoming duplicate physical IDs."""
    detection_box = tuple(detection["box"])
    detection_label = detection["object"]
    for track_id, track_box in track_boxes.items():
        if track_labels.get(track_id) != detection_label:
            continue
        detection_coverage, track_coverage = box_coverages(detection_box, track_box)
        if (
            box_iou(detection_box, track_box) >= minimum_iou
            or detection_coverage >= minimum_containment
            or track_coverage >= minimum_containment
        ):
            return True
    return False


def match_detections_to_tracks(
    track_boxes: dict[int, tuple[int, int, int, int]],
    detections: list[dict],
    minimum_iou: float = 0.30,
) -> tuple[dict[int, int], list[int]]:
    """Greedily assign unique high-IoU detections and return unmatched indices."""
    candidates = []
    for track_id, track_box in track_boxes.items():
        for detection_index, detection in enumerate(detections):
            overlap = box_iou(track_box, tuple(detection["box"]))
            if overlap >= minimum_iou:
                candidates.append((overlap, track_id, detection_index))

    assignments: dict[int, int] = {}
    used_detections = set()
    for _, track_id, detection_index in sorted(candidates, reverse=True):
        if track_id in assignments or detection_index in used_detections:
            continue
        assignments[track_id] = detection_index
        used_detections.add(detection_index)
    unmatched = [index for index in range(len(detections)) if index not in used_detections]
    return assignments, unmatched


def consensus_label(votes: Counter[str], newest_label: str) -> str:
    """Choose the vote leader, using newest evidence only to break a tie."""
    if not votes:
        return newest_label
    maximum = max(votes.values())
    leaders = {label for label, count in votes.items() if count == maximum}
    return newest_label if newest_label in leaders else sorted(leaders)[0]


def eligible_new_detection(
    detection: dict,
    frame_size: tuple[int, int],
    minimum_confidence: float,
) -> bool:
    """Reject weak, collapsed, and nearly full-frame unmatched detections."""
    if float(detection["confidence"]) < minimum_confidence:
        return False
    x1, y1, x2, y2 = detection["box"]
    width, height = frame_size
    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)
    area_ratio = (box_width * box_height) / max(1, width * height)
    return box_width >= 20 and box_height >= 20 and area_ratio <= 0.80


def requires_edge_safe_enrollment(label: str) -> bool:
    """Small portable objects must not be enrolled from edge-truncated crops."""
    return label in {
        "id card", "keys", "usb cable", "wired earphones",
        "wireless earbuds", "charger", "mobile phone", "remote control",
    }


def touches_frame_edge(
    box: tuple[int, int, int, int],
    frame_size: tuple[int, int],
    margin_ratio: float = 0.02,
) -> bool:
    width, height = frame_size
    margin_x = max(2, int(width * margin_ratio))
    margin_y = max(2, int(height * margin_ratio))
    x1, y1, x2, y2 = box
    return x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y


def tentative_track_action(
    confirmations: int,
    matched_same_label: bool,
    missed_refreshes: int,
    required_confirmations: int = 2,
    maximum_missed_refreshes: int = 0,
) -> str:
    """Return promote, keep, or discard for a provisional SAM 2 track."""
    if matched_same_label and confirmations >= required_confirmations:
        return "promote"
    if not matched_same_label and missed_refreshes > maximum_missed_refreshes:
        return "discard"
    return "keep"


def confirmed_track_action(
    matched: bool,
    missed_refreshes: int,
    maximum_missed_refreshes: int = 2,
) -> str:
    """Retire confirmed SAM state after sustained detector absence."""
    if matched:
        return "keep"
    return "retire" if missed_refreshes > maximum_missed_refreshes else "keep"
