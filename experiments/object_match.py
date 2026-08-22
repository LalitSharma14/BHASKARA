import cv2
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from ultralytics import SAM
from transformers import AutoImageProcessor, AutoModel


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

REFERENCE_PATH = (
    "reference_object/wired_earphones/"
    "wired_earphones_1.jpeg"
)

ROOM_PATH = "images/room.jpeg"


# --------------------------------------------------
# 2. LOAD DINOv2
# --------------------------------------------------
# DINOv2 will create visual embeddings for:
# - our reference earphones
# - every candidate object found by SAM
# --------------------------------------------------

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-small"
)

dino_model = AutoModel.from_pretrained(
    "facebook/dinov2-small"
)

# We are only using the model for prediction,
# not training it.
dino_model.eval()


# --------------------------------------------------
# 3. FUNCTION: IMAGE -> EMBEDDING
# --------------------------------------------------

def get_embedding(image):

    # Prepare PIL image for DINOv2
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # No gradient calculation needed
    with torch.no_grad():
        outputs = dino_model(**inputs)

    # CLS token represents the complete image/crop
    embedding = outputs.last_hidden_state[:, 0, :]

    # Normalize embedding.
    # This makes similarity comparison more stable.
    embedding = F.normalize(
        embedding,
        p=2,
        dim=1
    )

    return embedding


# --------------------------------------------------
# 4. LOAD REFERENCE EARPHONES
# --------------------------------------------------

reference_image = Image.open(
    REFERENCE_PATH
).convert("RGB")

reference_embedding = get_embedding(
    reference_image
)

print(
    "Reference embedding shape:",
    reference_embedding.shape
)


# --------------------------------------------------
# 5. LOAD ROOM IMAGE
# --------------------------------------------------

room_cv = cv2.imread(
    ROOM_PATH
)

if room_cv is None:
    print("Could not load room image")
    exit()

room_height, room_width = room_cv.shape[:2]


# --------------------------------------------------
# 6. LOAD SEGMENT ANYTHING MODEL
# --------------------------------------------------
# SAM does not tell us:
# "this is an earphone"
#
# Instead, it attempts to separate the image
# into individual candidate regions/objects.
# --------------------------------------------------

sam_model = SAM(
    "mobile_sam.pt"
)


# --------------------------------------------------
# 7. SEGMENT EVERYTHING IN THE ROOM
# --------------------------------------------------
# Because we are not providing points or boxes,
# SAM attempts automatic segmentation.
# --------------------------------------------------

sam_results = sam_model(
    ROOM_PATH,
    verbose=False
)


# We only supplied one image
sam_result = sam_results[0]


# --------------------------------------------------
# 8. CHECK WHETHER SEGMENTS WERE FOUND
# --------------------------------------------------

if sam_result.masks is None:

    print(
        "SAM did not find any candidate regions."
    )

    exit()


# --------------------------------------------------
# 9. GET MASKS AND THEIR BOUNDING BOXES
# --------------------------------------------------

masks = sam_result.masks.data.cpu().numpy()

boxes = sam_result.boxes.xyxy.cpu().numpy()

print(
    "Candidate objects found by SAM:",
    len(boxes)
)


# --------------------------------------------------
# 10. VARIABLES FOR BEST OBJECT MATCH
# --------------------------------------------------

best_score = -1

best_box = None

best_index = None


# --------------------------------------------------
# 11. CHECK EVERY CANDIDATE OBJECT
# --------------------------------------------------

for index, box in enumerate(boxes):

    # SAM box coordinates
    x1, y1, x2, y2 = map(
        int,
        box
    )


    # ----------------------------------------------
    # Ignore invalid boxes
    # ----------------------------------------------

    if x2 <= x1 or y2 <= y1:
        continue


    # ----------------------------------------------
    # Ignore extremely tiny segments
    #
    # These are often noise.
    # Keep threshold low because our target
    # object itself may be small.
    # ----------------------------------------------

    width = x2 - x1
    height = y2 - y1

    if width < 20 or height < 20:
        continue


    # ----------------------------------------------
    # Get this object's segmentation mask
    # ----------------------------------------------

    mask = masks[index]


    # Convert boolean/probability mask
    # into an 8-bit mask.
    mask = (
        mask > 0.5
    ).astype(np.uint8) * 255


    # Resize if required
    if (
        mask.shape[0] != room_height
        or mask.shape[1] != room_width
    ):

        mask = cv2.resize(
            mask,
            (room_width, room_height),
            interpolation=cv2.INTER_NEAREST
        )


    # ----------------------------------------------
    # Extract ONLY the segmented object
    # ----------------------------------------------

    segmented_object = cv2.bitwise_and(
        room_cv,
        room_cv,
        mask=mask
    )


    # Crop to object's bounding box
    crop = segmented_object[
        y1:y2,
        x1:x2
    ]


    if crop.size == 0:
        continue


    # ----------------------------------------------
    # OpenCV uses BGR.
    # PIL/DINO expects RGB.
    # ----------------------------------------------

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    crop_pil = Image.fromarray(
        crop_rgb
    )


    # ----------------------------------------------
    # Get candidate object's DINO embedding
    # ----------------------------------------------

    crop_embedding = get_embedding(
        crop_pil
    )


    # ----------------------------------------------
    # Compare candidate object with
    # registered wired earphones
    # ----------------------------------------------

    score = F.cosine_similarity(
        reference_embedding,
        crop_embedding
    ).item()


    print(
        "Candidate",
        index,
        "| Similarity:",
        round(score, 3),
        "| Box:",
        (x1, y1, x2, y2)
    )


    # ----------------------------------------------
    # Store best candidate
    # ----------------------------------------------

    if score > best_score:

        best_score = score

        best_box = (
            x1,
            y1,
            x2,
            y2
        )

        best_index = index


# --------------------------------------------------
# 12. DISPLAY BEST MATCH
# --------------------------------------------------

if best_box is None:

    print(
        "No suitable candidate object found."
    )

    exit()


x1, y1, x2, y2 = best_box


print()
print(
    "BEST MATCH"
)

print(
    "Candidate:",
    best_index
)

print(
    "Similarity:",
    best_score
)

print(
    "Box:",
    best_box
)


# Copy image so original stays unchanged
output_image = room_cv.copy()


# Draw best candidate box
cv2.rectangle(
    output_image,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3
)


# Label
label = (
    f"Wired Earphones "
    f"{best_score:.2f}"
)


cv2.putText(
    output_image,
    label,
    (
        x1,
        max(y1 - 10, 30)
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)


# --------------------------------------------------
# 13. SHOW RESULT
# --------------------------------------------------

cv2.imshow(
    "BHASKARA - Object Based Matching",
    output_image
)

cv2.waitKey(0)

cv2.destroyAllWindows()