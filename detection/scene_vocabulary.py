"""Dependency-light helpers for guarded dynamic scene vocabulary."""

from __future__ import annotations

import json
from pathlib import Path


PART_TO_OBJECT_LABELS = {
    "ceiling fan blades": "fan",
    "fan blades": "fan",
    "fan blade": "fan",
    "ceiling fan motor": "fan",
}


COMPONENT_LABELS = {"wheel"}


def normalize_scene_label(value: str) -> str | None:
    label = " ".join(str(value).lower().strip().split()).strip(" .,:;|[]()")
    if not label or len(label) > 60 or len(label.split()) > 4:
        return None
    return label


def load_scene_vocabulary(report_paths: list[str] | None, maximum_labels: int = 32) -> list[str]:
    """Load short Florence candidates; raw captions remain evidence, not prompts."""
    labels: list[str] = []
    for report_path in report_paths or []:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        for value in report.get("candidate_vocabulary", []):
            label = normalize_scene_label(value)
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= maximum_labels:
                return labels
    return labels


def canonical_scene_label(value: str, known_labels=()) -> str | None:
    """Reduce a dense caption to a short discovered label when possible."""
    raw = " ".join(str(value).lower().strip().split()).strip(" .,:;|[]()")
    if raw in PART_TO_OBJECT_LABELS:
        return PART_TO_OBJECT_LABELS[raw]
    candidates = sorted(
        (label for label in known_labels if label and label in raw),
        key=len,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    if " of " in raw:
        return normalize_scene_label(raw.split(" of ", 1)[0])
    return normalize_scene_label(raw)


def load_scene_proposals(report_paths: list[str] | None) -> dict[int, list[dict]]:
    """Load Florence boxes keyed by their exact evidence frame."""
    vocabulary = load_scene_vocabulary(report_paths)
    proposals: dict[int, list[dict]] = {}
    for report_path in report_paths or []:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        for frame in report.get("frames", []):
            frame_index = int(frame["frame_index"])
            for task, task_result in frame.get("tasks", {}).items():
                parsed = task_result.get("parsed", {}).get(task, {})
                for box, raw_label in zip(
                    parsed.get("bboxes", []), parsed.get("labels", [])
                ):
                    label = canonical_scene_label(raw_label, vocabulary)
                    if label is None or len(box) != 4:
                        continue
                    proposals.setdefault(frame_index, []).append({
                        "object": label,
                        "raw_object": raw_label,
                        "box": tuple(int(round(value)) for value in box),
                        "source": "florence",
                    })
    return proposals


def merge_search_classes(base_labels, dynamic_labels, maximum_dynamic: int = 32) -> list[str]:
    merged = list(dict.fromkeys(str(label).lower().strip() for label in base_labels))
    for value in dynamic_labels or []:
        label = normalize_scene_label(value)
        if label and label not in merged:
            merged.append(label)
        if len(merged) >= len(base_labels) + maximum_dynamic:
            break
    return merged


def dynamic_verification_candidates(label: str) -> tuple[str, ...]:
    """Return a small confusion policy for automatically discovered classes."""
    label = normalize_scene_label(label) or str(label).lower().strip()
    if label in {"home appliance", "household appliance", "wall appliance"}:
        alternatives = (
            "air conditioner", "heater", "refrigerator", "television",
            "electrical panel", "cabinet", "medicine box", label,
        )
    elif label == "fan":
        alternatives = ("fan", "ceiling fan", "ceiling light", "chandelier", "ceiling")
    elif "air conditioner" in label:
        alternatives = (label, "medicine box", "cabinet", "electrical panel")
    elif any(word in label for word in ("mirror", "painting", "picture frame")):
        alternatives = (label, "window", "mirror", "painting", "picture frame")
    else:
        alternatives = (label, "window", "cabinet", "clothes", "household object")
    return tuple(dict.fromkeys(alternatives))


def refined_scene_label(proposed_label: str, verification: dict | None) -> str | None:
    """Return a precise verified label, allowing only known generic classes to refine."""
    if not verification:
        return None
    proposed = normalize_scene_label(proposed_label) or str(proposed_label).lower().strip()
    winner = normalize_scene_label(verification.get("best_label", ""))
    if winner is None:
        return None
    if winner == proposed:
        # Generic labels are useful discovery hints but are too vague for
        # authoritative object memory.
        if proposed in {"home appliance", "household appliance", "wall appliance"}:
            return None
        return proposed
    if proposed in {"home appliance", "household appliance", "wall appliance"}:
        allowed = set(dynamic_verification_candidates(proposed)) - {proposed}
        return winner if winner in allowed else None
    return None


def possible_scene_label(proposed_label: str, raw_label: str) -> str | None:
    """Retain recognizable object-part evidence without trusting it as memory."""
    raw = " ".join(str(raw_label).lower().strip().split()).strip(" .,:;|[]()")
    canonical = canonical_scene_label(raw)
    if canonical == "fan" and any(part in raw for part in ("blade", "motor")):
        return "fan"
    if canonical in COMPONENT_LABELS:
        return canonical
    return None


def scene_label_requires_parent_context(label: str) -> bool:
    """Return whether a verified part must remain non-memory evidence initially."""
    normalized = normalize_scene_label(label) or str(label).lower().strip()
    return normalized in COMPONENT_LABELS


def _box_overlap(first, second) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return max(intersection / first_area, intersection / second_area)


def merge_verified_scene_proposals(
    detections: list[dict],
    proposals: list[dict],
    minimum_conflict_overlap: float = 0.65,
) -> list[dict]:
    """Replace overlapping detector guesses with semantically verified regions."""
    accepted: list[dict] = []
    for proposal in sorted(
        proposals, key=lambda item: float(item.get("confidence", 0.0)), reverse=True
    ):
        if any(
            _box_overlap(proposal["box"], existing["box"]) >= minimum_conflict_overlap
            for existing in accepted
        ):
            continue
        accepted.append(proposal)
    remaining = [
        detection
        for detection in detections
        if not any(
            _box_overlap(detection["box"], proposal["box"]) >= minimum_conflict_overlap
            for proposal in accepted
        )
    ]
    # Keep verified proposals first so a caller's object-count cap cannot drop
    # them behind ordinary detector guesses.
    return accepted + remaining
