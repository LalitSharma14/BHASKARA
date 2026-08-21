from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2


# Load Objects365 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26s-objv1-150.pt",
    confidence_threshold=0.40,
    device="cpu"
)


# Run sliced detection
result = get_sliced_prediction(
    "images/room.jpeg",
    detection_model,
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.20,
    overlap_width_ratio=0.20
)


# Read original image
image = cv2.imread("images/room.jpeg")


for prediction in result.object_prediction_list:

    object_name = prediction.category.name
    confidence = prediction.score.value

    x1, y1, x2, y2 = map(
        int,
        prediction.bbox.to_xyxy()
    )

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"{object_name} {confidence * 100:.1f}%"

    # Find size of label text
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )

    # Position label
    label_y = max(y1, text_height + 10)

    # Draw filled background behind label
    cv2.rectangle(
        image,
        (x1, label_y - text_height - 8),
        (x1 + text_width + 8, label_y),
        (0, 255, 0),
        -1
    )

    # Draw label text
    cv2.putText(
        image,
        label,
        (x1 + 4, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

    print(
        object_name,
        round(confidence * 100, 2),
        "%"
    )


cv2.imshow(
    "BHASKARA - Clean Sliced Detection",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()