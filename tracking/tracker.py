# --------------------------------------------------
# BHASKARA
# Object Tracking Module
# --------------------------------------------------

from detection.detector import (
    model,
    FINDABLE_OBJECTS,
    LOCATION_OBJECTS,
    ALLOWED_CLASSES
)


# --------------------------------------------------
# 1. Track objects in a video frame
# --------------------------------------------------

def track_objects(frame):
    """
    Detect and track objects in a video frame.

    Returns:
        A list of dictionaries containing:

        object      -> object name
        confidence  -> detection confidence
        box         -> bounding box
        type        -> findable / location
        track_id    -> unique tracking ID
    """


    # --------------------------------------------------
    # 2. Run YOLO-World + ByteTrack
    # --------------------------------------------------

    results = model.track(
        frame,

        # Keep tracking information between frames
        persist=True,

        # Use ByteTrack
        tracker="bytetrack.yaml",

        # Detection settings
        conf=0.15,
        imgsz=960,
        iou=0.35,

        verbose=False
    )


    result = results[0]

    tracked_objects = []


    # --------------------------------------------------
    # 3. Process every detected/tracked object
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


        # --------------------------------------------------
        # Ignore classes BHASKARA doesn't care about
        # --------------------------------------------------

        if object_name not in ALLOWED_CLASSES:
            continue


        # --------------------------------------------------
        # Different confidence requirements
        # --------------------------------------------------

        
        # --------------------------------------------------
        # Different confidence requirements
        # --------------------------------------------------

        if object_name in FINDABLE_OBJECTS:

            # Small/findable objects are often harder to detect,
            # so we allow a lower confidence threshold.
            min_confidence = 0.25


        elif object_name in [
            "room door",
            "glass window",
            "cabinet",
            "shelf"
        ]:

            # Structural objects such as doors, windows,
            # cabinets and shelves are visually similar.
            # Require stronger confidence before accepting them.
            min_confidence = 0.55


        else:

            # Other large location objects
            min_confidence = 0.40


        # Ignore detection if confidence is too low
        if confidence < min_confidence:
            continue

        

        # --------------------------------------------------
        # Bounding box
        # --------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        width = x2 - x1
        height = y2 - y1


        if width < 15 or height < 15:
            continue


        # --------------------------------------------------
        # 4. Get tracking ID
        # --------------------------------------------------

        # ByteTrack attaches an ID to tracked objects.
        # On some first-frame detections ID can be None.
        if box.id is not None:

            track_id = int(
                box.id[0]
            )

        else:

            track_id = -1


        # --------------------------------------------------
        # Normalize descriptive YOLO-World names
        # --------------------------------------------------

        if object_name == "room door":
            final_name = "door"

        elif object_name == "glass window":
            final_name = "window"

        else:
            final_name = object_name


        # --------------------------------------------------
        # Find object type
        # --------------------------------------------------

        if object_name in FINDABLE_OBJECTS:

            object_type = "findable"

        else:

            object_type = "location"


        # --------------------------------------------------
        # 5. Save tracked object
        # --------------------------------------------------

        tracked_objects.append({

            "object": final_name,

            "confidence": confidence,

            "box": (
                x1,
                y1,
                x2,
                y2
            ),

            "type": object_type,

            "track_id": track_id
        })


    return tracked_objects