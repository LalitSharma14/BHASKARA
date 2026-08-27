"""Crop-quality and geometry helpers for visual object identity."""


def clamp_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    return (
        max(0, min(int(x1), frame_width - 1)),
        max(0, min(int(y1), frame_height - 1)),
        max(0, min(int(x2), frame_width)),
        max(0, min(int(y2), frame_height)),
    )


def describe_crop(box, frame_width, frame_height):
    """Describe whether a detection crop is reliable enough for identity."""

    x1, y1, x2, y2 = clamp_box(box, frame_width, frame_height)
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    frame_area = max(1, frame_width * frame_height)
    area_ratio = (width * height) / frame_area
    aspect_ratio = width / height if height else 0.0
    touches_edge = x1 == 0 or y1 == 0 or x2 == frame_width or y2 == frame_height

    # Very small crops are dominated by resize artifacts. Edge crops are still
    # useful, but receive a lower reliability score because the object may be
    # only partially visible.
    valid = width >= 16 and height >= 16 and area_ratio >= 0.0003
    quality = min(1.0, min(width, height) / 48.0)
    if touches_edge:
        quality *= 0.75

    return {
        "box": (x1, y1, x2, y2),
        "width": width,
        "height": height,
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "touches_edge": touches_edge,
        "quality": quality,
        "valid": valid,
        "tiny": min(width, height) < 32 or area_ratio < 0.001,
    }


def geometry_is_compatible(observation, identity, minimum_aspect_ratio=0.35):
    """Reject severe shape changes while allowing viewpoint and scale changes."""

    new_aspect = observation.get("appearance_aspect_ratio")
    old_aspect = identity.get("appearance_aspect_ratio")
    if not new_aspect or not old_aspect:
        return True

    ratio = min(new_aspect, old_aspect) / max(new_aspect, old_aspect)
    return ratio >= minimum_aspect_ratio


def required_identity_similarity(observation, base_similarity):
    """Demand stronger appearance evidence for tiny or partial observations."""

    threshold = base_similarity
    if observation.get("appearance_tiny"):
        threshold = max(threshold, 0.96)
    if observation.get("appearance_touches_edge"):
        threshold = max(threshold, 0.95)
    return threshold
