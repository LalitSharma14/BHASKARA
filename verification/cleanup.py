# --------------------------------------------------
# BHASKARA
# Final Detection Cleanup
# --------------------------------------------------

from verification.verifier import verify_candidates


# --------------------------------------------------
# 1. Confusion groups
# --------------------------------------------------

CONFUSION_GROUPS = {
    "door": ["door", "window", "cabinet"],
    "window": ["window", "door", "cabinet"],
    "cabinet": ["cabinet", "door", "window"],

    "wired earphones": [
        "wired earphones",
        "wireless earbuds",
        "usb cable",
        "charger"
    ],

    "charger": [
        "charger",
        "usb cable",
        "wired earphones"
    ],

    "mobile phone": [
        "mobile phone",
        "mouse",
        "remote control"
    ],

    "bottle": [
        "bottle",
        "cup"
    ]
}


# --------------------------------------------------
# 2. Normalize strange compound labels
# --------------------------------------------------

def normalize_label(label):

    text = label.lower().strip()

    if "wired earphones" in text:
        return "wired earphones"

    if "wireless earbuds" in text:
        return "wireless earbuds"

    if "usb cable" in text:
        return "usb cable"

    if "charger" in text:
        return "charger"

    if "mobile phone" in text:
        return "mobile phone"

    if "mouse" in text:
        return "mouse"

    if "remote control" in text:
        return "remote control"

    if "metal ruler" in text:
        return "metal ruler"

    if "glasses" in text:
        return "glasses"

    if "keys" in text:
        return "keys"

    if "bottle" in text:
        return "bottle"

    if "cup" in text:
        return "cup"

    if "bed" in text:
        return "bed"

    if "sofa" in text:
        return "sofa"

    if "chair" in text:
        return "chair"

    if "desk" in text:
        return "desk"

    if "door" in text:
        return "door"

    if "window" in text:
        return "window"

    if "cabinet" in text:
        return "cabinet"

    return text


# --------------------------------------------------
# 3. IoU helper
# --------------------------------------------------

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        (box1[2] - box1[0])
        * (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0])
        * (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


# --------------------------------------------------
# 4. Remove duplicate final detections
# --------------------------------------------------

def remove_duplicates(
    detections,
    iou_threshold=0.35,
    containment_threshold=0.60
):
    """
    Remove repeated detections of the same object.

    Uses:
    1. IoU overlap
    2. How much the smaller box is contained inside
       the larger box

    This is especially useful for long/thin objects
    such as cables, rulers and earphones.
    """

    # Highest-confidence detections first
    detections = sorted(
        detections,
        key=lambda x: x["confidence"],
        reverse=True
    )

    final_detections = []

    for detection in detections:

        duplicate = False

        box1 = detection["box"]

        for existing in final_detections:

            # Only compare detections that ended up
            # with the same final label
            if detection["object"] != existing["object"]:
                continue

            box2 = existing["box"]


            # ------------------------------------------
            # Calculate intersection
            # ------------------------------------------

            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])

            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])


            intersection_width = max(
                0,
                x2 - x1
            )

            intersection_height = max(
                0,
                y2 - y1
            )

            intersection = (
                intersection_width
                * intersection_height
            )


            # ------------------------------------------
            # Areas
            # ------------------------------------------

            area1 = (
                (box1[2] - box1[0])
                * (box1[3] - box1[1])
            )

            area2 = (
                (box2[2] - box2[0])
                * (box2[3] - box2[1])
            )


            if area1 <= 0 or area2 <= 0:
                continue


            # ------------------------------------------
            # Standard IoU
            # ------------------------------------------

            union = (
                area1
                + area2
                - intersection
            )

            iou = (
                intersection / union
                if union > 0
                else 0
            )


            # ------------------------------------------
            # How much of the SMALLER box overlaps
            # the other box
            # ------------------------------------------

            smaller_area = min(
                area1,
                area2
            )

            containment = (
                intersection
                / smaller_area
            )


            # ------------------------------------------
            # Treat as duplicate if:
            #
            # boxes overlap strongly
            # OR
            # one box is mostly inside another
            # ------------------------------------------

            if (
                iou >= iou_threshold
                or containment >= containment_threshold
            ):

                duplicate = True
                break


        if not duplicate:

            final_detections.append(
                detection
            )

    return final_detections