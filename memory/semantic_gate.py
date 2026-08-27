"""Pure semantic admission rules for trusted object memory."""


SEMANTIC_MEMORY_REQUIREMENTS = {
    "charger": {"minimum_score": 0.40, "minimum_margin": 0.10},
    "usb cable": {"minimum_score": 0.40, "minimum_margin": 0.10},
    "wired earphones": {"minimum_score": 0.40, "minimum_margin": 0.10},
}


def semantic_memory_decision(track):
    """Return whether a track has adequate semantic evidence for memory.

    Tracks outside the high-risk confusion set retain the existing trust path.
    A successful decision is cached on the track. Rejections are not permanent:
    later detector evidence can still correct or strengthen the classification.
    """

    label = track["object"]
    requirement = SEMANTIC_MEMORY_REQUIREMENTS.get(label)
    if requirement is None:
        return {"accepted": True, "reason": "not_required"}

    if (
        track.get("semantic_memory_approved")
        and track.get("semantic_memory_label") == label
        and track.get("semantic_verified_label") == label
    ):
        return {"accepted": True, "reason": "cached"}

    verified_label = track.get("semantic_verified_label")
    score = track.get("semantic_score")
    margin = track.get("semantic_margin")

    if verified_label != label or score is None or margin is None:
        return {"accepted": False, "reason": "missing_or_conflicting_evidence"}
    if score < requirement["minimum_score"]:
        return {"accepted": False, "reason": "low_score"}
    if margin < requirement["minimum_margin"]:
        return {"accepted": False, "reason": "ambiguous"}

    track["semantic_memory_approved"] = True
    track["semantic_memory_label"] = label
    return {"accepted": True, "reason": "verified"}
