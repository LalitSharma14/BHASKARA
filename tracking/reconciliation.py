"""Pure helpers for associating fresh detections with active tracks.

The matcher uses conservative identity gates before calculating a score. This
keeps plausible label corrections possible without allowing overlap alone to
merge semantically unrelated objects.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.appearance_quality import (
    geometry_is_compatible,
    required_identity_similarity,
)


COMPATIBLE_LABEL_GROUPS = (
    frozenset({"bed", "desk"}),
)


def calculate_iou(box1, box2):
    """Return intersection-over-union for two ``(x1, y1, x2, y2)`` boxes."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def calculate_containment(box1, box2):
    """Return the intersection divided by the smaller box's area."""

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    if area1 <= 0 or area2 <= 0:
        return 0.0

    return intersection / min(area1, area2)


def get_box_center(box):
    """Return the center point of a bounding box."""

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_box_area(box):
    """Return the non-negative area of a bounding box."""

    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def get_box_diagonal(box):
    """Return a box's diagonal length."""

    width = max(0, box[2] - box[0])
    height = max(0, box[3] - box[1])
    return (width * width + height * height) ** 0.5


def are_labels_compatible(old_label, new_label):
    """Return whether two labels may describe the same physical object."""

    if old_label == new_label:
        return True

    label_pair = frozenset({old_label, new_label})
    return any(label_pair <= group for group in COMPATIBLE_LABEL_GROUPS)


def select_consensus_label(label_votes, newest_label):
    """Select the majority label, preferring newest evidence on an exact tie."""

    if not label_votes:
        return newest_label

    return max(
        label_votes,
        key=lambda label: (
            label_votes[label],
            label == newest_label,
        ),
    )


def cosine_similarity(embedding1, embedding2):
    """Return cosine similarity, or ``None`` when appearance is unavailable."""

    if embedding1 is None or embedding2 is None:
        return None

    vector1 = np.asarray(embedding1, dtype=np.float32).reshape(-1)
    vector2 = np.asarray(embedding2, dtype=np.float32).reshape(-1)

    if vector1.shape != vector2.shape or vector1.size == 0:
        return None

    denominator = float(np.linalg.norm(vector1) * np.linalg.norm(vector2))
    if denominator <= 0:
        return None

    return float(np.dot(vector1, vector2) / denominator)


def update_appearance_prototype(previous, observation, observation_weight=0.10):
    """Update a normalized appearance prototype without allowing abrupt drift."""

    if observation is None:
        return previous

    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    observation_norm = float(np.linalg.norm(observation))
    if observation_norm <= 0:
        return previous
    observation = observation / observation_norm

    if previous is None:
        return observation

    previous = np.asarray(previous, dtype=np.float32).reshape(-1)
    if previous.shape != observation.shape:
        return observation

    combined = (
        previous * (1.0 - observation_weight)
        + observation * observation_weight
    )
    combined_norm = float(np.linalg.norm(combined))
    return combined / combined_norm if combined_norm > 0 else observation


def update_appearance_gallery(
    gallery,
    observation,
    maximum_size=8,
    diversity_threshold=0.98,
):
    """Keep a compact set of normalized, visually diverse observations."""

    updated_gallery = list(gallery or [])
    normalized_observation = update_appearance_prototype(None, observation)
    if normalized_observation is None:
        return updated_gallery

    similarities = [
        cosine_similarity(normalized_observation, prototype)
        for prototype in updated_gallery
    ]
    similarities = [value for value in similarities if value is not None]

    if similarities and max(similarities) >= diversity_threshold:
        return updated_gallery

    updated_gallery.append(normalized_observation)
    if len(updated_gallery) > maximum_size:
        updated_gallery = updated_gallery[-maximum_size:]

    return updated_gallery


def best_appearance_similarity(observation, identity):
    """Compare an observation with every available identity prototype."""

    if observation is None:
        return None

    prototypes = list(identity.get("appearance_gallery", []) or [])
    running_prototype = identity.get("appearance_embedding")
    if running_prototype is not None:
        prototypes.append(running_prototype)

    similarities = [
        cosine_similarity(observation, prototype)
        for prototype in prototypes
    ]
    similarities = [value for value in similarities if value is not None]
    return max(similarities) if similarities else None


def identity_appearance_similarity(observation, identity):
    """Return object-focused similarity with context used only as support.

    A high context score can never rescue a weak tight-object score. This is
    important in a single-room video where unrelated objects share a bed,
    wall, or floor background. Context also never penalizes a strong object
    match because it changes substantially as the camera viewpoint changes.
    """

    tight_similarity = best_appearance_similarity(
        observation.get("appearance_embedding"),
        identity,
    )
    if tight_similarity is None:
        return None

    context_identity = {
        "appearance_embedding": identity.get("context_embedding"),
        "appearance_gallery": identity.get("context_gallery", []),
    }
    context_similarity = best_appearance_similarity(
        observation.get("context_embedding"),
        context_identity,
    )
    if context_similarity is None:
        return tight_similarity

    context_support = max(0.0, context_similarity - tight_similarity)
    return min(1.0, tight_similarity + context_support * 0.05)


def calculate_identity_score(detection, track, frame_diagonal):
    """Combine spatial/motion evidence with crop appearance evidence."""

    spatial_score = calculate_match_score(
        detection["box"],
        track["box"],
        detection["object"],
        track["object"],
        frame_diagonal,
    )

    if spatial_score <= 0:
        return 0.0

    if not geometry_is_compatible(detection, track):
        return 0.0

    appearance = identity_appearance_similarity(detection, track)

    if appearance is None:
        return spatial_score

    minimum_similarity = (
        0.50
        if detection["object"] == track["object"]
        else 0.70
    )

    if appearance < minimum_similarity:
        return 0.0

    return spatial_score * 0.55 + max(0.0, appearance) * 0.45


def calculate_match_score(new_box, old_box, new_label, old_label, frame_diagonal):
    """Calculate a gated, scale-aware physical-identity score.

    ``frame_diagonal`` remains in the signature for compatibility with the
    video pipeline. Object-relative distance is used because a fixed fraction
    of the full frame is too permissive for small indoor objects.
    """

    del frame_diagonal

    if not are_labels_compatible(old_label, new_label):
        return 0.0

    iou = calculate_iou(new_box, old_box)
    containment = calculate_containment(new_box, old_box)

    new_area = get_box_area(new_box)
    old_area = get_box_area(old_box)
    if new_area <= 0 or old_area <= 0:
        return 0.0

    area_ratio = min(new_area, old_area) / max(new_area, old_area)

    new_center = get_box_center(new_box)
    old_center = get_box_center(old_box)
    dx = new_center[0] - old_center[0]
    dy = new_center[1] - old_center[1]
    center_distance = (dx * dx + dy * dy) ** 0.5

    reference_diagonal = max(
        get_box_diagonal(new_box),
        get_box_diagonal(old_box),
    )

    if reference_diagonal <= 0:
        return 0.0

    normalized_distance = center_distance / reference_diagonal

    same_label = old_label == new_label

    if same_label:
        spatially_plausible = (
            iou >= 0.10
            or (
                containment >= 0.60
                and normalized_distance <= 0.65
            )
        )
    else:
        spatially_plausible = (
            (
                iou >= 0.15
                and normalized_distance <= 0.55
            )
            or (
                containment >= 0.75
                and normalized_distance <= 0.35
                and area_ratio >= 0.15
            )
        )

    if not spatially_plausible:
        return 0.0

    proximity = max(0.0, 1.0 - normalized_distance)

    return (
        iou * 0.55
        + containment * 0.20
        + area_ratio * 0.15
        + proximity * 0.10
    )


def find_best_track(
    tracks,
    used_track_ids,
    new_box,
    new_label,
    frame_diagonal,
    is_valid_box,
):
    """Return the highest-scoring unused track and its current score."""

    best_track = None
    best_score = 0.0

    for track in tracks:
        if track["id"] in used_track_ids:
            continue

        old_box = track["box"]
        if not is_valid_box(old_box):
            continue

        score = calculate_match_score(
            new_box,
            old_box,
            new_label,
            track["object"],
            frame_diagonal,
        )

        if score > best_score:
            best_score = score
            best_track = track

    return best_track, best_score


def find_global_assignments(
    detections,
    tracks,
    frame_diagonal,
    is_valid_box,
    match_threshold=0.30,
):
    """Return order-independent, one-to-one detection/track assignments.

    Every eligible pair is scored first. Pairs are then accepted from strongest
    to weakest while ensuring that each detection and track is used at most
    once. The result maps detection indexes to ``(track, score)`` tuples.
    """

    if not detections or not tracks:
        return {}

    score_matrix = np.zeros(
        (len(detections), len(tracks)),
        dtype=np.float32,
    )

    for detection_index, detection in enumerate(detections):
        new_box = detection["box"]
        if not is_valid_box(new_box):
            continue

        for track_index, track in enumerate(tracks):
            old_box = track["box"]
            if not is_valid_box(old_box):
                continue

            score = calculate_identity_score(
                detection,
                track,
                frame_diagonal,
            )

            if score >= match_threshold:
                score_matrix[detection_index, track_index] = score

    detection_indexes, track_indexes = linear_sum_assignment(
        -score_matrix
    )

    assignments = {}
    for detection_index, track_index in zip(
        detection_indexes,
        track_indexes,
    ):
        score = float(score_matrix[detection_index, track_index])
        if score >= match_threshold:
            assignments[int(detection_index)] = (
                tracks[int(track_index)],
                score,
            )

    return assignments


def find_lost_track_assignments(
    detections,
    lost_tracks,
    frame_diagonal,
    is_valid_box,
    minimum_similarity=0.92,
    minimum_margin=0.03,
    return_diagnostics=False,
):
    """Find conservative recovery matches for recently lost tracks.

    Recovery requires the exact same consensus label, strong appearance, and a
    clear margin over the second-best identity candidate.
    """

    candidates = []
    similarities_by_detection = {}
    diagnostics = {
        "same_label_comparisons": 0,
        "above_threshold": 0,
        "ambiguous": 0,
        "assigned": 0,
        "best_similarities": [],
    }

    for detection_index, detection in enumerate(detections):
        if not is_valid_box(detection["box"]):
            continue

        for track in lost_tracks:
            if detection["object"] != track["object"]:
                continue
            if not is_valid_box(track["box"]):
                continue

            diagnostics["same_label_comparisons"] += 1

            if not geometry_is_compatible(detection, track):
                continue

            appearance = identity_appearance_similarity(detection, track)

            if appearance is None:
                continue

            similarities_by_detection.setdefault(
                detection_index,
                []
            ).append(appearance)

            effective_minimum = required_identity_similarity(
                detection,
                minimum_similarity,
            )
            if appearance < effective_minimum:
                continue

            diagnostics["above_threshold"] += 1

            spatial_score = calculate_match_score(
                detection["box"],
                track["box"],
                detection["object"],
                track["object"],
                frame_diagonal,
            )

            recovery_score = appearance * 0.85 + spatial_score * 0.15
            candidates.append(
                (recovery_score, appearance, detection_index, track)
            )

    diagnostics["best_similarities"] = [
        max(similarities)
        for similarities in similarities_by_detection.values()
        if similarities
    ]

    unambiguous_candidates = []
    for detection_index in range(len(detections)):
        detection_candidates = [
            candidate
            for candidate in candidates
            if candidate[2] == detection_index
        ]
        detection_candidates.sort(key=lambda item: item[1], reverse=True)

        if not detection_candidates:
            continue

        best = detection_candidates[0]
        if len(detection_candidates) > 1:
            appearance_margin = best[1] - detection_candidates[1][1]
            if appearance_margin < minimum_margin:
                diagnostics["ambiguous"] += 1
                continue

        unambiguous_candidates.append(best)

    unambiguous_candidates.sort(key=lambda item: item[0], reverse=True)
    assignments = {}
    used_track_ids = set()

    for score, _appearance, detection_index, track in unambiguous_candidates:
        if detection_index in assignments or track["id"] in used_track_ids:
            continue
        assignments[detection_index] = (track, score)
        used_track_ids.add(track["id"])

    diagnostics["assigned"] = len(assignments)

    if return_diagnostics:
        return assignments, diagnostics

    return assignments
