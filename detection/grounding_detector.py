# --------------------------------------------------
# BHASKARA
# Grounding DINO Detector
# --------------------------------------------------

import cv2
import torch
import numpy as np

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)

from torchvision.ops import nms


# --------------------------------------------------
# 1. Model configuration
# --------------------------------------------------

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# 2. Load Grounding DINO
# --------------------------------------------------

print("Loading Grounding DINO...")


processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_NAME
)


model = model.to(device)

model.eval()


print("Grounding DINO loaded.")


# --------------------------------------------------
# 3. Broad household vocabulary
# --------------------------------------------------

SEARCH_CLASSES = [
    "wired earphones",
    "wireless earbuds",
    "glasses",
    "keys",
    "charger",
    "usb cable",
    "metal ruler",
    "wallet",
    "remote control",
    "mobile phone",
    "bottle",
    "book",
    "backpack",
    "watch",
    "shoes",
    "medicine box",
    "id card",
    "pen",
    "scissors",
    "cup",
    "mouse",
    "keyboard",
    "clothes",

    # Location / room objects
    "bed",
    "chair",
    "table",
    "desk",
    "door",
    "window",
    "cabinet",
    "shelf",
    "fan",
    "sofa",
    "pillow",
    "television",
    "laptop"
]


# --------------------------------------------------
# 4. Main detection function
# --------------------------------------------------

def detect_objects(
    image_input,
    box_threshold=0.20,
    text_threshold=0.20,
    nms_threshold=0.40,
    search_classes=None,
):
    """
    Run Grounding DINO on an image.

    image_input can be:

    1. File path
       "images/room.jpeg"

    2. PIL Image

    3. OpenCV / NumPy frame
       frame returned from video.read()

    Returns:

    [
        {
            "object": "glasses",
            "confidence": 0.79,
            "box": (x1, y1, x2, y2)
        }
    ]
    """


    # --------------------------------------------------
    # 5. Load / prepare input image
    # --------------------------------------------------

    # Case 1: image path
    if isinstance(image_input, str):

        image = Image.open(
            image_input
        ).convert("RGB")


    # Case 2: PIL image
    elif isinstance(image_input, Image.Image):

        image = image_input.convert(
            "RGB"
        )


    # Case 3: OpenCV video frame / NumPy array
    elif isinstance(image_input, np.ndarray):

        # OpenCV stores frames in BGR format
        # Grounding DINO expects RGB
        image_rgb = cv2.cvtColor(
            image_input,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            image_rgb
        )


    # Unsupported input
    else:

        raise TypeError(
            "detect_objects() expects an image path, "
            "PIL Image, or OpenCV NumPy frame"
        )


    # --------------------------------------------------
    # 6. Create Grounding DINO text prompt
    # --------------------------------------------------

    active_search_classes = search_classes or SEARCH_CLASSES
    text_prompt = (
        ". ".join(active_search_classes)
        + "."
    )


    # --------------------------------------------------
    # 7. Prepare image and text
    # --------------------------------------------------

    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt"
    )


    # Move tensors to CPU/GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------
    # 8. Run Grounding DINO inference
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )


    # --------------------------------------------------
    # 9. Convert model output into image coordinates
    # --------------------------------------------------

    target_sizes = torch.tensor(
        [
            image.size[::-1]
        ]
    ).to(device)


    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        target_sizes=target_sizes,
        threshold=box_threshold,
        text_threshold=text_threshold
    )


    result = results[0]


    # --------------------------------------------------
    # 10. Extract outputs
    # --------------------------------------------------

    boxes = result["boxes"]

    scores = result["scores"]

    labels = result.get(
        "text_labels",
        result["labels"]
    )


    # --------------------------------------------------
    # 11. No detections
    # --------------------------------------------------

    if len(boxes) == 0:

        return []


    # --------------------------------------------------
    # 12. NMS
    # --------------------------------------------------
    # Removes strongly overlapping duplicate boxes.
    # --------------------------------------------------

    keep_indices = nms(
        boxes,
        scores,
        nms_threshold
    )


    detections = []


    # --------------------------------------------------
    # 13. Build clean detection list
    # --------------------------------------------------

    for index in keep_indices:

        index = int(index)


        box = boxes[index]

        confidence = float(
            scores[index]
        )

        object_name = str(
            labels[index]
        )


        x1, y1, x2, y2 = map(
            int,
            box.tolist()
        )


        # --------------------------------------------------
        # Skip invalid boxes
        # --------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            continue


        # --------------------------------------------------
        # Save detection
        # --------------------------------------------------

        detections.append({

            "object": object_name,

            "confidence": confidence,

            "box": (
                x1,
                y1,
                x2,
                y2
            )
        })


    # --------------------------------------------------
    # 14. Return detections
    # --------------------------------------------------

    return detections
