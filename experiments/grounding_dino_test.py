# --------------------------------------------------
# BHASKARA
# Grounding DINO Open-Vocabulary Detection Test
# --------------------------------------------------

import cv2
import torch

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)


# --------------------------------------------------
# 1. Model name
# --------------------------------------------------

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"


# --------------------------------------------------
# 2. Test image
# --------------------------------------------------

IMAGE_PATH = "images/room.jpeg"


# --------------------------------------------------
# 3. Objects we want to search for
# --------------------------------------------------
# Grounding DINO is open-vocabulary, so these
# descriptions are not limited to a fixed COCO list.
# --------------------------------------------------

TEST_CLASSES = [
    "wired earphones",
    "glasses",
    "metal ruler",
    "keys",
    "charger",
    "backpack",
    "bed",
    "remote control",
    "bottle",
    "laptop"
]


# --------------------------------------------------
# 4. Select device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# --------------------------------------------------
# 5. Load processor and Grounding DINO model
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

print("Grounding DINO loaded successfully.")


# --------------------------------------------------
# 6. Load image
# --------------------------------------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# --------------------------------------------------
# 7. Prepare image and text prompts
# --------------------------------------------------

inputs = processor(
    images=image,
    text=TEST_CLASSES,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# --------------------------------------------------
# 8. Run inference
# --------------------------------------------------

with torch.no_grad():

    outputs = model(
        **inputs
    )


# --------------------------------------------------
# 9. Convert raw outputs into boxes and labels
# --------------------------------------------------

target_sizes = torch.tensor(
    [image.size[::-1]]
).to(device)


results = processor.post_process_grounded_object_detection(
    outputs,
    input_ids=inputs["input_ids"],
    target_sizes=target_sizes,

    # Box confidence threshold
    threshold=0.20,

    # Text-label matching threshold
    text_threshold=0.20
)


result = results[0]


# --------------------------------------------------
# 10. Load image with OpenCV for drawing
# --------------------------------------------------

image_cv = cv2.imread(
    IMAGE_PATH
)

if image_cv is None:
    print("Could not load image with OpenCV.")
    exit()


# --------------------------------------------------
# 11. Print number of detections
# --------------------------------------------------

print(
    "\nNumber of detections:",
    len(result["boxes"])
)

print()


# --------------------------------------------------
# 12. Draw every detection
# --------------------------------------------------

for box, score, label in zip(
    result["boxes"],
    result["scores"],
    result["labels"]
):

    # Convert bounding box into integers
    x1, y1, x2, y2 = map(
        int,
        box.tolist()
    )


    confidence = float(
        score
    )


    object_name = str(
        label
    )


    # --------------------------------------------------
    # Print detection
    # --------------------------------------------------

    print(
        object_name,
        "-",
        round(
            confidence * 100,
            2
        ),
        "%",
        "| Box:",
        (x1, y1, x2, y2)
    )


    # --------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------

    cv2.rectangle(
        image_cv,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # --------------------------------------------------
    # Create label
    # --------------------------------------------------

    text = (
        f"{object_name} "
        f"{confidence * 100:.1f}%"
    )


    # --------------------------------------------------
    # Draw label
    # --------------------------------------------------

    cv2.putText(
        image_cv,
        text,
        (
            x1,
            max(y1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )


# --------------------------------------------------
# 13. Show final image
# --------------------------------------------------

cv2.imshow(
    "BHASKARA - Grounding DINO Test",
    image_cv
)

cv2.waitKey(0)

cv2.destroyAllWindows()


print("\nGrounding DINO test finished.")