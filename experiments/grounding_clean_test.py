# --------------------------------------------------
# BHASKARA
# Grounding DINO + SigLIP + Final Cleanup
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
# 1. Test image
# --------------------------------------------------

IMAGE_PATH = "images/room.jpeg"


# --------------------------------------------------
# 2. Known labels used to split compound phrases
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
# 3. Extract possible labels from compound text
# --------------------------------------------------

def extract_candidate_labels(compound_label):

    text = compound_label.lower().strip()

    candidates = []

    for known_label in KNOWN_LABELS:

        if known_label in text:

            candidates.append(
                known_label
            )


    # Handle vague words that Grounding DINO
    # sometimes returns
    if "earphones" in text and "wired earphones" not in candidates:

        candidates.append(
            "wired earphones"
        )


    if "cable" in text and "usb cable" not in candidates:

        candidates.append(
            "usb cable"
        )


    if "control" in text and "remote control" not in candidates:

        candidates.append(
            "remote control"
        )


    if not candidates:

        candidates.append(
            normalize_label(text)
        )


    # Remove duplicates
    return list(
        dict.fromkeys(candidates)
    )


# --------------------------------------------------
# 4. Run Grounding DINO
# --------------------------------------------------

detections = detect_objects(
    IMAGE_PATH,
    box_threshold=0.20,
    text_threshold=0.20,
    nms_threshold=0.40
)


# --------------------------------------------------
# 5. Load original image
# --------------------------------------------------

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    print("Could not load image.")

    exit()


print(
    "\nBHASKARA FINAL VISION PIPELINE TEST"
)

print(
    "------------------------------------------"
)

print(
    "Grounding DINO detections:",
    len(detections)
)


# --------------------------------------------------
# 6. Final results will be stored here
# --------------------------------------------------

final_detections = []


# --------------------------------------------------
# 7. Process every detection
# --------------------------------------------------

for detection in detections:

    grounding_label = detection["object"]

    grounding_confidence = detection[
        "confidence"
    ]

    x1, y1, x2, y2 = detection[
        "box"
    ]


    print()
    print(
        "------------------------------------------"
    )

    print(
        "Grounding DINO:",
        grounding_label
    )


    # --------------------------------------------------
    # 8. Candidate labels
    # --------------------------------------------------

    candidates = extract_candidate_labels(
        grounding_label
    )


    print(
        "Candidates:",
        candidates
    )


    # Default label
    final_name = normalize_label(
        grounding_label
    )


    # --------------------------------------------------
    # 9. Crop candidate region
    # --------------------------------------------------

    height, width = image.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1


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
        width,
        x2 + pad_x
    )

    crop_y2 = min(
        height,
        y2 + pad_y
    )


    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


    # --------------------------------------------------
    # 10. Verify if multiple meanings exist
    # --------------------------------------------------

    if (
        crop.size != 0
        and len(candidates) > 1
    ):

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

            print(
                "SigLIP scores:"
            )


            for label, score in verification[
                "scores"
            ].items():

                print(
                    " ",
                    label,
                    ":",
                    round(
                        score * 100,
                        2
                    ),
                    "%"
                )


            final_name = verification[
                "best_label"
            ]


            print(
                "SigLIP winner:",
                final_name
            )


    else:

        print(
            "SigLIP skipped."
        )


    # --------------------------------------------------
    # 11. Normalize final label
    # --------------------------------------------------

    final_name = normalize_label(
        final_name
    )


    # --------------------------------------------------
    # 12. Store final detection
    # --------------------------------------------------

    final_detections.append({

        "object": final_name,

        "confidence": grounding_confidence,

        "box": (
            x1,
            y1,
            x2,
            y2
        )
    })


# --------------------------------------------------
# 13. Remove duplicate final detections
# --------------------------------------------------

final_detections = remove_duplicates(
    final_detections,
    iou_threshold=0.45
)


# --------------------------------------------------
# 14. Print final clean results
# --------------------------------------------------

print()
print(
    "=========================================="
)

print(
    "FINAL CLEAN DETECTIONS:",
    len(final_detections)
)

print(
    "=========================================="
)


for detection in final_detections:

    print(
        detection["object"],
        "-",
        round(
            detection["confidence"] * 100,
            2
        ),
        "%",
        "| Box:",
        detection["box"]
    )


# --------------------------------------------------
# 15. Draw ONLY final detections
# --------------------------------------------------

output_image = image.copy()


for detection in final_detections:

    object_name = detection[
        "object"
    ]

    confidence = detection[
        "confidence"
    ]

    x1, y1, x2, y2 = detection[
        "box"
    ]


    # Draw box
    cv2.rectangle(
        output_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # Label
    label = (
        f"{object_name} "
        f"{confidence * 100:.0f}%"
    )


    # Calculate text size
    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )


    label_y = max(
        y1,
        text_height + 10
    )


    # Filled background
    cv2.rectangle(
        output_image,
        (
            x1,
            label_y - text_height - 8
        ),
        (
            x1 + text_width + 8,
            label_y
        ),
        (0, 255, 0),
        -1
    )


    # Text
    cv2.putText(
        output_image,
        label,
        (
            x1 + 4,
            label_y - 4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )


# --------------------------------------------------
# 16. Show final cleaned image
# --------------------------------------------------

cv2.imshow(
    "BHASKARA - Final Vision Pipeline",
    output_image
)

cv2.waitKey(0)

cv2.destroyAllWindows()


print(
    "\nFinal vision pipeline test finished."
)