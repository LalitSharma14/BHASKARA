# --------------------------------------------------
# BHASKARA
# Final Vision Detector
#
# Pipeline:
# Grounding DINO
#      ↓
# Candidate label extraction
#      ↓
# SigLIP verification
#      ↓
# Label normalization
#      ↓
# Duplicate cleanup
#      ↓
# Final clean detections
# --------------------------------------------------

import cv2

from PIL import Image

from detection.grounding_detector import detect_objects
from verification.verifier import verify_candidates
from verification.cleanup import (
    normalize_label,
    remove_duplicates
)


# --------------------------------------------------
# 1. Known labels
# --------------------------------------------------

KNOWN_LABELS = [
    "wired earphones",
    "wireless earbuds",
    "usb cable",
    "mobile phone",
    "remote control",
    "metal ruler",
    "medicine box",
    "glasses",
    "keys",
    "charger",
    "mouse",
    "bed",
    "sofa",
    "bottle",
    "cup",
    "wallet",
    "door",
    "window",
    "cabinet",
    "chair",
    "desk",
    "pen",
    "scissors",
    "book",
    "pillow"
]


# --------------------------------------------------
# 2. Objects users may want to find
# --------------------------------------------------

FINDABLE_OBJECTS = {
    "wired earphones",
    "wireless earbuds",
    "usb cable",
    "mobile phone",
    "remote control",
    "metal ruler",
    "medicine box",
    "glasses",
    "keys",
    "charger",
    "mouse",
    "bottle",
    "cup",
    "wallet",
    "pen",
    "scissors",
    "book"
}


# --------------------------------------------------
# 3. Extract possible meanings from
#    Grounding DINO compound labels
# --------------------------------------------------

def extract_candidate_labels(compound_label):

    text = compound_label.lower().strip()

    candidates = []


    # Match known labels inside the Grounding DINO text
    for known_label in KNOWN_LABELS:

        if known_label in text:
            candidates.append(
                known_label
            )


    # --------------------------------------------------
    # Handle vague / partial words
    # --------------------------------------------------

    if (
        "earphones" in text
        and "wired earphones" not in candidates
    ):
        candidates.append(
            "wired earphones"
        )


    if (
        "cable" in text
        and "usb cable" not in candidates
    ):
        candidates.append(
            "usb cable"
        )


    if (
        "control" in text
        and "remote control" not in candidates
    ):
        candidates.append(
            "remote control"
        )


    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    if not candidates:

        candidates.append(
            normalize_label(text)
        )


    # Remove duplicate labels while preserving order
    return list(
        dict.fromkeys(candidates)
    )


# --------------------------------------------------
# 4. Main detector
# --------------------------------------------------

def detect_final_objects(
    image_input,
    box_threshold=0.20,
    text_threshold=0.20,
    nms_threshold=0.40
):
    """
    Run BHASKARA's final vision pipeline.

    image_input can be:

    1. Image path:
       "images/room.jpeg"

    2. OpenCV video frame:
       NumPy array returned by video.read()

    Returns:

    [
        {
            "object": "glasses",
            "confidence": 0.79,
            "box": (x1, y1, x2, y2),
            "type": "findable"
        }
    ]
    """


    # --------------------------------------------------
    # 5. Prepare OpenCV image
    #    We need this for cropping and SigLIP
    # --------------------------------------------------

    if isinstance(image_input, str):

        image = cv2.imread(
            image_input
        )

        if image is None:

            print(
                "Could not load image:",
                image_input
            )

            return []


    else:

        # image_input is assumed to be an OpenCV frame
        image = image_input.copy()


    # --------------------------------------------------
    # 6. Run Grounding DINO
    #
    # grounding_detector.py already supports:
    # - image paths
    # - PIL images
    # - OpenCV / NumPy frames
    # --------------------------------------------------

    detections = detect_objects(
        image_input,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_threshold=nms_threshold
    )


    # Get image dimensions
    image_height, image_width = image.shape[:2]


    verified_detections = []


    # --------------------------------------------------
    # 7. Process each Grounding DINO detection
    # --------------------------------------------------

    for detection in detections:

        grounding_label = detection[
            "object"
        ]

        confidence = detection[
            "confidence"
        ]

        x1, y1, x2, y2 = detection[
            "box"
        ]


        # --------------------------------------------------
        # 8. Extract candidate object names
        # --------------------------------------------------

        candidates = extract_candidate_labels(
            grounding_label
        )


        # Default label before SigLIP
        final_name = normalize_label(
            grounding_label
        )


        # --------------------------------------------------
        # 9. Prepare crop for SigLIP
        # --------------------------------------------------

        box_width = x2 - x1
        box_height = y2 - y1


        # Slight padding around the detected box
        pad_x = int(
            box_width * 0.05
        )

        pad_y = int(
            box_height * 0.05
        )


        crop_x1 = max(
            0,
            x1 - pad_x
        )

        crop_y1 = max(
            0,
            y1 - pad_y
        )

        crop_x2 = min(
            image_width,
            x2 + pad_x
        )

        crop_y2 = min(
            image_height,
            y2 + pad_y
        )


        crop = image[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]


        # --------------------------------------------------
        # 10. Run SigLIP verification
        #
        # Only needed when Grounding DINO returns
        # multiple possible meanings.
        # --------------------------------------------------

        if (
            crop.size != 0
            and len(candidates) > 1
        ):

            # OpenCV uses BGR
            # SigLIP/PIL expects RGB
            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )


            crop_pil = Image.fromarray(
                crop_rgb
            )


            verification = verify_candidates(
                crop_pil,
                candidates
            )


            if verification is not None:

                final_name = verification[
                    "best_label"
                ]


        # --------------------------------------------------
        # 11. Normalize final label
        # --------------------------------------------------

        final_name = normalize_label(
            final_name
        )


        # --------------------------------------------------
        # 12. Decide object type
        # --------------------------------------------------

        if final_name in FINDABLE_OBJECTS:

            object_type = "findable"

        else:

            object_type = "location"


        # --------------------------------------------------
        # 13. Store verified detection
        # --------------------------------------------------

        verified_detections.append({

            "object": final_name,

            "confidence": confidence,

            "box": (
                x1,
                y1,
                x2,
                y2
            ),

            "type": object_type
        })


    # --------------------------------------------------
    # 14. Remove duplicate final detections
    # --------------------------------------------------

    final_detections = remove_duplicates(
        verified_detections,
        iou_threshold=0.35,
        containment_threshold=0.60
    )


    # --------------------------------------------------
    # 15. Return final results
    # --------------------------------------------------

    return final_detections