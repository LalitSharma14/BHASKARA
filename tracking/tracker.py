# --------------------------------------------------
# BHASKARA
# Tracking Module
#
# Pipeline:
# final_detector.py
#      ↓
# Grounding DINO + SigLIP
#      ↓
# clean detections
#      ↓
# ByteTrack
#      ↓
# persistent track IDs
# --------------------------------------------------

import numpy as np
import supervision as sv

from detection.final_detector import detect_final_objects


# --------------------------------------------------
# 1. Create ByteTrack tracker
# --------------------------------------------------

tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.80,
    frame_rate=30
)


# --------------------------------------------------
# 2. Track one video frame
# --------------------------------------------------

def track_objects(frame):
    """
    Detect and track objects in a video frame.

    Returns:

    [
        {
            "object": "glasses",
            "confidence": 0.79,
            "box": (x1, y1, x2, y2),
            "type": "findable",
            "track_id": 3
        }
    ]
    """


    # --------------------------------------------------
    # 3. Run BHASKARA final detector
    # --------------------------------------------------

    detections = detect_final_objects(
        frame
    )


    # No detections
    if len(detections) == 0:

        return []


    # --------------------------------------------------
    # 4. Convert BHASKARA detections
    #    into arrays ByteTrack understands
    # --------------------------------------------------

    boxes = []

    confidences = []

    class_ids = []


    # We need our own mapping:
    #
    # object name -> integer ID
    # --------------------------------------------------

    object_names = []

    name_to_id = {}


    for detection in detections:

        object_name = detection[
            "object"
        ]


        # Create numeric class ID
        if object_name not in name_to_id:

            name_to_id[
                object_name
            ] = len(name_to_id)


        class_id = name_to_id[
            object_name
        ]


        boxes.append(
            detection["box"]
        )

        confidences.append(
            detection["confidence"]
        )

        class_ids.append(
            class_id
        )

        object_names.append(
            object_name
        )


    # --------------------------------------------------
    # 5. Create Supervision Detections object
    # --------------------------------------------------

    sv_detections = sv.Detections(

        xyxy=np.array(
            boxes,
            dtype=np.float32
        ),

        confidence=np.array(
            confidences,
            dtype=np.float32
        ),

        class_id=np.array(
            class_ids,
            dtype=np.int32
        )
    )


    # --------------------------------------------------
    # 6. Run ByteTrack
    # --------------------------------------------------

    tracked = tracker.update_with_detections(
        sv_detections
    )


    # --------------------------------------------------
    # 7. Build final tracked-object list
    # --------------------------------------------------

    tracked_objects = []


    for i in range(
        len(tracked)
    ):

        x1, y1, x2, y2 = map(
            int,
            tracked.xyxy[i]
        )


        confidence = float(
            tracked.confidence[i]
        )


        class_id = int(
            tracked.class_id[i]
        )


        track_id = int(
            tracked.tracker_id[i]
        )


        # --------------------------------------------------
        # Recover object name
        # --------------------------------------------------

        object_name = None


        for name, numeric_id in name_to_id.items():

            if numeric_id == class_id:

                object_name = name
                break


        # --------------------------------------------------
        # Find original object type
        # --------------------------------------------------

        object_type = "location"


        for detection in detections:

            if detection["object"] == object_name:

                object_type = detection[
                    "type"
                ]

                break


        # --------------------------------------------------
        # Save tracked object
        # --------------------------------------------------

        tracked_objects.append({

            "object": object_name,

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