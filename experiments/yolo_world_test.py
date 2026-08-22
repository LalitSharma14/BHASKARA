from ultralytics import YOLOWorld
import cv2


# --------------------------------------------------
# 1. Load YOLO-World model
# --------------------------------------------------

model = YOLOWorld("yolov8s-world.pt")


# --------------------------------------------------
# 2. Tell YOLO-World what objects to search for
# --------------------------------------------------

model.set_classes([
    "bed",
    "laptop",
    "charger",
    "wired earphones",
    "keys",
    ""
])


# --------------------------------------------------
# 3. Classes we actually want to keep
# --------------------------------------------------

allowed_classes = {
    "bed",
    "laptop",
    "charger",
    "wired earphones",
    "keys"
}


# --------------------------------------------------
# 4. Run detection
# --------------------------------------------------
# conf=0.15 removes very weak detections
# imgsz=1280 helps small-object detection
# iou=0.45 controls duplicate-box suppression
# --------------------------------------------------

results = model.predict(
    "images/room.jpeg",
    conf=0.15,
    imgsz=1280,
    iou=0.45
)


# We supplied only one image
result = results[0]


# --------------------------------------------------
# 5. Print number of raw detections
# --------------------------------------------------

print(
    "Raw detections:",
    len(result.boxes)
)


# --------------------------------------------------
# 6. Load original room image
# --------------------------------------------------

image = cv2.imread(
    "images/room.jpeg"
)

if image is None:
    print("Could not load room image")
    exit()


# Count only detections that survive our filters
final_detection_count = 0


# --------------------------------------------------
# 7. Go through every detection
# --------------------------------------------------

for box in result.boxes:

    # Get class ID
    class_id = int(
        box.cls[0]
    )

    # Get object name
    object_name = result.names[
        class_id
    ]

    # Get confidence
    confidence = float(
        box.conf[0]
    )


    # --------------------------------------------------
    # FILTER 1:
    # Remove classes we do not care about
    # --------------------------------------------------

    if object_name not in allowed_classes:
        continue


    # --------------------------------------------------
    # FILTER 2:
    # Remove detections below 20% confidence
    # --------------------------------------------------

    if confidence < 0.20:
        continue


    # --------------------------------------------------
    # Get bounding-box coordinates
    # --------------------------------------------------

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )


    # --------------------------------------------------
    # FILTER 3:
    # Ignore extremely tiny boxes
    # --------------------------------------------------
    # Sometimes very small false positives appear.
    # Keep this low because keys/earphones are small.
    # --------------------------------------------------

    width = x2 - x1
    height = y2 - y1

    if width < 15 or height < 15:
        continue


    # Detection survived all filters
    final_detection_count += 1


    # --------------------------------------------------
    # Print detection
    # --------------------------------------------------

    print(
        object_name,
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
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # --------------------------------------------------
    # Create label
    # --------------------------------------------------

    label = (
        f"{object_name} "
        f"{confidence * 100:.1f}%"
    )


    # --------------------------------------------------
    # Calculate label size
    # --------------------------------------------------

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        1
    )

    label_y = max(
        y1,
        text_height + 10
    )


    # --------------------------------------------------
    # Draw filled background behind label
    # --------------------------------------------------

    cv2.rectangle(
        image,
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


    # --------------------------------------------------
    # Draw label text
    # --------------------------------------------------

    cv2.putText(
        image,
        label,
        (
            x1 + 4,
            label_y - 4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1
    )


# --------------------------------------------------
# 8. Print final count
# --------------------------------------------------

print(
    "Final detections after filtering:",
    final_detection_count
)


# --------------------------------------------------
# 9. Show final result
# --------------------------------------------------

cv2.imshow(
    "BHASKARA - Filtered YOLO World",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()