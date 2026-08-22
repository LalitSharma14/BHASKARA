from ultralytics import YOLOWorld


# --------------------------------------------------
# 1. Load YOLO-World model once
# --------------------------------------------------

model = YOLOWorld("yolov8s-world.pt")


# --------------------------------------------------
# 2. Define the classes BHASKARA currently cares about
# --------------------------------------------------

SEARCH_CLASSES = [
    "bed",
    "laptop",
    "charger",
    "wired earphones",
    "keys",
    ""
]


# Tell YOLO-World which classes to search for
model.set_classes(SEARCH_CLASSES)


# --------------------------------------------------
# 3. Classes we actually want returned
# --------------------------------------------------

ALLOWED_CLASSES = {
    "bed",
    "laptop",
    "charger",
    "wired earphones",
    "keys"
}


# --------------------------------------------------
# 4. Reusable detection function
# --------------------------------------------------

def detect_objects(image_path):
    """
    Detect objects in an image using YOLO-World.

    Returns a list of dictionaries.
    Each dictionary contains:
    - object name
    - confidence
    - bounding box
    """

    # Run YOLO-World detection
    results = model.predict(
        image_path,
        conf=0.15,
        imgsz=1280,
        iou=0.45,
        verbose=False
    )

    # We are currently processing one image at a time
    result = results[0]

    detections = []


    # --------------------------------------------------
    # Process every prediction
    # --------------------------------------------------

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )

        object_name = result.names[
            class_id
        ]

        confidence = float(
            box.conf[0]
        )


        # ----------------------------------------------
        # Filter unwanted classes
        # ----------------------------------------------

        if object_name not in ALLOWED_CLASSES:
            continue


        # ----------------------------------------------
        # Filter weak detections
        # ----------------------------------------------

        if confidence < 0.20:
            continue


        # ----------------------------------------------
        # Get bounding-box coordinates
        # ----------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        width = x2 - x1
        height = y2 - y1


        # ----------------------------------------------
        # Ignore extremely tiny detections
        # ----------------------------------------------

        if width < 15 or height < 15:
            continue


        # ----------------------------------------------
        # Store clean detection
        # ----------------------------------------------

        detection = {
            "object": object_name,
            "confidence": confidence,
            "box": (
                x1,
                y1,
                x2,
                y2
            )
        }

        detections.append(
            detection
        )


    return detections