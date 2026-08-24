# --------------------------------------------------
# BHASKARA
# SigLIP 2 Fine-Grained Object Verifier
# --------------------------------------------------

import torch
from PIL import Image

from transformers import (
    AutoModel,
    AutoProcessor
)


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Verifier device:", device)


# --------------------------------------------------
# 2. Load SigLIP 2
# --------------------------------------------------

MODEL_NAME = "google/siglip2-base-patch16-224"

print("Loading SigLIP 2 verifier...")


processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

model.eval()


print("SigLIP 2 loaded successfully.")


# --------------------------------------------------
# 3. Convert simple labels into better descriptions
# --------------------------------------------------
# Vision-language models often work better with
# natural descriptions than single words.
# --------------------------------------------------

LABEL_DESCRIPTIONS = {

    "wired earphones":
        "a pair of wired earphones with visible earbuds and wire",

    "wireless earbuds":
        "a pair of small wireless earbuds",

    "charger":
        "a phone charger or charging adapter",

    "usb cable":
        "a USB charging or data cable",

    "mobile phone":
        "a mobile smartphone",

    "mouse":
        "a computer mouse",

    "bed":
        "a bed used for sleeping",

    "sofa":
        "a sofa or couch",

    "bottle":
        "a drinking bottle",

    "cup":
        "a drinking cup",

    "medicine box":
        "a small medicine box or medicine container",

    "keys":
        "a set of metal keys",

    "glasses":
        "a pair of eyeglasses",

    "metal ruler":
        "a long metal measuring ruler",

    "remote control":
        "a handheld remote control",

    "wallet":
        "a small personal wallet",

    "door":
        "a room door with a handle",

    "window":
        "a glass window",

    "cabinet":
        "a storage cabinet",

    "chair":
        "a chair for sitting",

    "desk":
        "a desk or work table",

    "pen":
        "a writing pen",

    "scissors":
        "a pair of scissors",

    "book":
        "a book",

    "pillow":
        "a pillow"
}


# --------------------------------------------------
# 4. Convert label -> description
# --------------------------------------------------

def get_description(label):

    label = label.lower().strip()

    return LABEL_DESCRIPTIONS.get(
        label,
        f"a photo of {label}"
    )


# --------------------------------------------------
# 5. Main verification function
# --------------------------------------------------

def verify_candidates(
    image,
    candidate_labels
):
    """
    Ask SigLIP which candidate label best matches
    the supplied object crop.

    image:
        PIL Image OR image path

    candidate_labels:
        Example:
        ["charger", "medicine box"]

    Returns:
        {
            "best_label": "charger",
            "best_score": 0.82,
            "scores": {
                "charger": 0.82,
                "medicine box": 0.18
            }
        }
    """


    # --------------------------------------------------
    # Validate candidates
    # --------------------------------------------------

    if not candidate_labels:
        return None


    # Remove duplicate labels while preserving order
    candidate_labels = list(
        dict.fromkeys(
            label.lower().strip()
            for label in candidate_labels
        )
    )


    # --------------------------------------------------
    # If only one possibility exists,
    # verification isn't necessary
    # --------------------------------------------------

    if len(candidate_labels) == 1:

        return {
            "best_label": candidate_labels[0],
            "best_score": 1.0,
            "scores": {
                candidate_labels[0]: 1.0
            }
        }


    # --------------------------------------------------
    # Load image if a path was supplied
    # --------------------------------------------------

    if isinstance(image, str):

        image = Image.open(
            image
        ).convert("RGB")

    else:

        image = image.convert("RGB")


    # --------------------------------------------------
    # Convert candidate names to descriptions
    # --------------------------------------------------

    descriptions = [
        get_description(label)
        for label in candidate_labels
    ]


    # --------------------------------------------------
    # Prepare SigLIP input
    # --------------------------------------------------

    inputs = processor(
        text=descriptions,
        images=image,
        padding="max_length",
        return_tensors="pt"
    )


    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------
    # Run verifier
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )


    logits = outputs.logits_per_image[0]


    # Relative comparison among candidates
    probabilities = torch.softmax(
        logits,
        dim=0
    )


    # --------------------------------------------------
    # Build readable scores
    # --------------------------------------------------

    scores = {}


    for label, probability in zip(
        candidate_labels,
        probabilities
    ):

        scores[label] = float(
            probability.cpu()
        )


    # --------------------------------------------------
    # Find winner
    # --------------------------------------------------

    best_label = max(
        scores,
        key=scores.get
    )


    return {

        "best_label": best_label,

        "best_score": scores[best_label],

        "scores": scores
    }