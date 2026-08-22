import torch
import torch.nn.functional as F
import cv2
import numpy as np

from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# --------------------------------------------------
# 1. Load DINOv2 image processor and model
# --------------------------------------------------

processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-small"
)

model = AutoModel.from_pretrained(
    "facebook/dinov2-small"
)

# Put model in inference mode
model.eval()


# --------------------------------------------------
# 2. Load reference image and room image
# --------------------------------------------------

reference_image = Image.open(
    "reference_object/wired_earphones/wired_earphones_1.jpeg"
).convert("RGB")

room_image = Image.open(
    "images/room.jpeg"
).convert("RGB")


# --------------------------------------------------
# 3. Function to convert an image into an embedding
# --------------------------------------------------

def get_embedding(image):

    # Convert the image into the format expected by DINOv2
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Disable gradient calculations because we are not training
    with torch.no_grad():

        outputs = model(**inputs)

    # Extract the CLS token embedding
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding


# --------------------------------------------------
# 4. Function to compare two embeddings
# --------------------------------------------------

def cosine_similarity(a, b):

    # Higher value = more visually similar
    return F.cosine_similarity(
        a,
        b
    ).item()


# --------------------------------------------------
# 5. Generate embedding for reference earphones
# --------------------------------------------------

reference_embedding = get_embedding(
    reference_image
)

print(
    "Reference embedding shape:",
    reference_embedding.shape
)


# --------------------------------------------------
# 6. Generate embedding for full room image
#    This is only for verification
# --------------------------------------------------

room_embedding = get_embedding(
    room_image
)

print(
    "Room embedding shape:",
    room_embedding.shape
)


# --------------------------------------------------
# 7. Sliding window configuration
# --------------------------------------------------

# Size of each region that we will inspect
tile_size = 128

# Movement of the window each time
# 112 gives approximately 50% overlap
step = 32


# --------------------------------------------------
# 8. Variables to store the best matching region
# --------------------------------------------------

best_score = -1

best_box = None


# --------------------------------------------------
# 9. Get room dimensions
# --------------------------------------------------

room_width, room_height = room_image.size


# --------------------------------------------------
# 10. Slide a box across the entire room image
# --------------------------------------------------

for y in range(
    0,
    room_height - tile_size + 1,
    step
):

    for x in range(
        0,
        room_width - tile_size + 1,
        step
    ):

        # Crop a small region from the room image
        crop = room_image.crop(
            (
                x,
                y,
                x + tile_size,
                y + tile_size
            )
        )


        # Convert this crop into a DINOv2 embedding
        crop_embedding = get_embedding(
            crop
        )


        # Compare crop with reference earphones
        score = cosine_similarity(
            reference_embedding,
            crop_embedding
        )


        # If this region is more similar than
        # everything checked before, save it
        if score > best_score:

            best_score = score

            best_box = (
                x,
                y,
                x + tile_size,
                y + tile_size
            )


# --------------------------------------------------
# 11. Print final result
# --------------------------------------------------

print(
    "Best similarity:",
    best_score
)

print(
    "Best box:",
    best_box
)

import cv2
import numpy as np


# Convert PIL room image to OpenCV format
room_cv = cv2.cvtColor(
    np.array(room_image),
    cv2.COLOR_RGB2BGR
)


# Get coordinates of best matching box
x1, y1, x2, y2 = best_box


# Draw bounding box
cv2.rectangle(
    room_cv,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3
)


# Create label
label = f"Wired Earphones {best_score:.2f}"


# Draw label
cv2.putText(
    room_cv,
    label,
    (x1, max(y1 - 10, 25)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)


# Show result
cv2.imshow(
    "BHASKARA - Embedding Match",
    room_cv
)

cv2.waitKey(0)
cv2.destroyAllWindows()