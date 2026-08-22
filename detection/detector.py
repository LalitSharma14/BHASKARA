from ultralytics import YOLOWorld


# --------------------------------------------------
# 1. Load YOLO-World model
# --------------------------------------------------

model = YOLOWorld("yolov8s-world.pt")


# --------------------------------------------------
# 2. Objects users may want to find
# --------------------------------------------------

FINDABLE_OBJECTS = [
    "keys",
    "wallet",
    "glasses",
    "wired earphones",
    "wireless earbuds",
    "charger",
    "mobile phone",
    "remote control",
    "bottle",
    "book",
    "backpack",
    "watch",
    "shoes",
    "medicine box",
    "USB drive",
    "ID card",
    "pen",
    "scissors",
    "cup",
    "mouse",
    "keyboard",
    "clothes"
]


# --------------------------------------------------
# 3. Objects useful for describing location
# --------------------------------------------------

LOCATION_OBJECTS = [
    "bed",
    "chair",
    "table",
    "desk",
    "room door",
    "glass window",
    "shelf",
    "cabinet",
    "fan",
    "sofa",
    "pillow",
    "blanket",
    "television",
    "laptop"
]


# --------------------------------------------------
# 4. Complete search vocabulary
# --------------------------------------------------

SEARCH_CLASSES = (
    FINDABLE_OBJECTS
    + LOCATION_OBJECTS
    + [""]
)

model.set_classes(SEARCH_CLASSES)


# --------------------------------------------------
# 5. Valid classes
# --------------------------------------------------

ALLOWED_CLASSES = set(
    FINDABLE_OBJECTS
    + LOCATION_OBJECTS
)


# --------------------------------------------------
# 6. Reusable detection function
# --------------------------------------------------

def detect_objects(image):

    results = model.predict(
        image,
        conf=0.15,
        imgsz=960,
        iou=0.35,
        verbose=False
    )

    result = results[0]

    detections = []


    for box in result.boxes:

        class_id = int(box.cls[0])

        object_name = result.names[class_id]

        confidence = float(box.conf[0])


        # ----------------------------------------------
        # Ignore unwanted classes
        # ----------------------------------------------

        if object_name not in ALLOWED_CLASSES:
            continue


        # ----------------------------------------------
        # Different confidence thresholds
        # ----------------------------------------------

        if object_name in FINDABLE_OBJECTS:
            min_confidence = 0.25

        else:
            min_confidence = 0.40


        if confidence < min_confidence:
            continue


        # ----------------------------------------------
        # Get bounding box
        # ----------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        width = x2 - x1
        height = y2 - y1


        # Ignore extremely tiny detections
        if width < 15 or height < 15:
            continue


        # ----------------------------------------------
        # Normalize descriptive prompt names
        # ----------------------------------------------

        if object_name == "room door":
            final_name = "door"

        elif object_name == "glass window":
            final_name = "window"

        else:
            final_name = object_name


        # ----------------------------------------------
        # Determine object type
        # ----------------------------------------------

        if object_name in FINDABLE_OBJECTS:
            object_type = "findable"

        else:
            object_type = "location"


        detections.append({
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


    return detections