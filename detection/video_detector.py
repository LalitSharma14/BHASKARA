# --------------------------------------------------
# BHASKARA
#
# Grounding DINO
# + Selective SigLIP
# + Asynchronous Detection
# + Optical Flow
# + Track Reconciliation
# + Label Voting
# + Trusted Memory
# --------------------------------------------------

import cv2
import threading
import numpy as np

from PIL import Image

from detection.grounding_detector import detect_objects
from verification.verifier import verify_candidates

from memory.object_memory import (
    update_memory,
    get_memory
)


# ==================================================
# CONFIGURATION
# ==================================================

VIDEO_PATH = "videos/room.mp4"


# Run Grounding DINO periodically
PROCESS_EVERY = 5


# Object must be independently detected this
# many times before entering memory
MIN_CONFIRMATIONS = 3


# Keep missed tracks alive briefly
MAX_MISSED_SCANS = 2


# Reject collapsed boxes
MIN_BOX_WIDTH = 10
MIN_BOX_HEIGHT = 10


# ==================================================
# OBJECT VOCABULARY
# ==================================================

FINDABLE_OBJECTS = {

    "wired earphones",
    "wireless earbuds",

    "glasses",
    "keys",

    "charger",
    "usb cable",

    "metal ruler",

    "wallet",
    "remote control",
    "mobile phone",

    "bottle",
    "book",
    "watch",

    "medicine box",
    "id card",

    "pen",
    "scissors",

    "cup",

    "mouse",
    "keyboard",

    "clothes"
}


LOCATION_OBJECTS = {

    "bed",
    "chair",
    "table",
    "desk",

    "door",
    "window",

    "cabinet",
    "shelf",

    "sofa",

    "laptop",

    "fan"
}


ALLOWED_OBJECTS = (
    FINDABLE_OBJECTS
    | LOCATION_OBJECTS
)


# ==================================================
# CLASS-SPECIFIC CONFIDENCE THRESHOLDS
# ==================================================

CLASS_THRESHOLDS = {

    # Small / difficult objects

    "keys": 0.20,

    "wired earphones": 0.24,
    "wireless earbuds": 0.24,

    "usb cable": 0.24,

    "glasses": 0.24,

    "clothes": 0.22,

    "metal ruler": 0.25,

    "pen": 0.22,

    "scissors": 0.24,

    "id card": 0.22,


    # Common findable objects

    "charger": 0.28,

    "wallet": 0.28,

    "mobile phone": 0.28,

    "remote control": 0.28,

    "bottle": 0.28,

    "book": 0.25,

    "cup": 0.28,

    "mouse": 0.28,

    "keyboard": 0.30,

    "medicine box": 0.28,

    "watch": 0.25,


    # Larger objects

    "bed": 0.38,

    "chair": 0.35,

    "table": 0.35,

    "desk": 0.35,

    "sofa": 0.38,

    "laptop": 0.30,


    # Structural objects

    "door": 0.42,

    "window": 0.42,

    "cabinet": 0.42,

    "shelf": 0.40,

    "fan": 0.30
}


DEFAULT_THRESHOLD = 0.30


# ==================================================
# SIGLIP CONFUSION GROUPS
# ==================================================

CONFUSION_GROUPS = {

    "charger": [
        "charger",
        "usb cable",
        "wired earphones",
        "fan"
    ],

    "usb cable": [
        "usb cable",
        "charger",
        "wired earphones",
        "fan"
    ],

    "wired earphones": [
        "wired earphones",
        "usb cable",
        "charger"
    ],

    "door": [
        "door",
        "window",
        "cabinet"
    ],

    "window": [
        "window",
        "door",
        "cabinet"
    ],

    "cabinet": [
        "cabinet",
        "door",
        "window"
    ]
}


# ==================================================
# VIDEO
# ==================================================

video = cv2.VideoCapture(
    VIDEO_PATH
)


if not video.isOpened():

    print(
        "Could not open video."
    )

    raise SystemExit


fps = video.get(
    cv2.CAP_PROP_FPS
)


if fps <= 0:
    fps = 30


delay = max(
    1,
    int(1000 / fps)
)


print(
    "Video FPS:",
    round(fps, 2)
)


# ==================================================
# GLOBAL STATE
# ==================================================

frame_number = 0


detection_running = False

pending_detections = None


lock = threading.Lock()


tracks = []

next_track_id = 1


previous_gray = None


# ==================================================
# LABEL NORMALIZATION
# ==================================================

def normalize_live_label(label):

    text = label.lower().strip()


    ordered_labels = [

        "wired earphones",
        "wireless earbuds",

        "remote control",
        "mobile phone",

        "metal ruler",
        "medicine box",

        "usb cable",

        "glasses",
        "charger",
        "keys",
        "wallet",

        "bottle",
        "book",
        "watch",

        "id card",

        "scissors",
        "keyboard",
        "mouse",

        "clothes",
        "fan",

        "pen",
        "cup",

        "laptop",

        "cabinet",
        "window",
        "door",

        "desk",
        "table",
        "chair",

        "shelf",
        "sofa",
        "bed"
    ]


    for object_name in ordered_labels:

        if object_name in text:

            return object_name


    return text


# ==================================================
# CONFIDENCE FILTER
# ==================================================

def passes_threshold(
    object_name,
    confidence
):

    threshold = CLASS_THRESHOLDS.get(
        object_name,
        DEFAULT_THRESHOLD
    )


    return confidence >= threshold


# ==================================================
# BOX VALIDATION
# ==================================================

def is_valid_box(
    box,
    frame_width=None,
    frame_height=None
):

    x1, y1, x2, y2 = box


    width = x2 - x1

    height = y2 - y1


    if width < MIN_BOX_WIDTH:
        return False


    if height < MIN_BOX_HEIGHT:
        return False


    if x2 <= x1 or y2 <= y1:
        return False


    if frame_width is not None:

        if x1 >= frame_width:
            return False

        if x2 <= 0:
            return False


    if frame_height is not None:

        if y1 >= frame_height:
            return False

        if y2 <= 0:
            return False


    return True


# ==================================================
# IoU
# ==================================================

def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )


    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )


    intersection = (
        intersection_width
        * intersection_height
    )


    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )


    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )


    union = (
        area1
        + area2
        - intersection
    )


    if union <= 0:
        return 0.0


    return intersection / union


# ==================================================
# CONTAINMENT
#
# Useful when:
#
# partial bed -> detected as desk
# later full bed -> detected as bed
#
# Standard IoU may be low even though the first
# box is mostly inside the second.
# ==================================================

def calculate_containment(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )


    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )


    intersection = (
        intersection_width
        * intersection_height
    )


    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )


    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )


    if area1 <= 0 or area2 <= 0:
        return 0.0


    smaller_area = min(
        area1,
        area2
    )


    return (
        intersection
        / smaller_area
    )


# ==================================================
# BOX CENTER
# ==================================================

def get_box_center(box):

    x1, y1, x2, y2 = box


    center_x = (
        x1 + x2
    ) / 2


    center_y = (
        y1 + y2
    ) / 2


    return (
        center_x,
        center_y
    )


# ==================================================
# OPTICAL FLOW FEATURE POINTS
# ==================================================

def get_points_inside_box(
    gray_image,
    box
):

    x1, y1, x2, y2 = box


    height, width = gray_image.shape[:2]


    x1 = max(
        0,
        min(
            int(x1),
            width - 1
        )
    )


    x2 = max(
        0,
        min(
            int(x2),
            width
        )
    )


    y1 = max(
        0,
        min(
            int(y1),
            height - 1
        )
    )


    y2 = max(
        0,
        min(
            int(y2),
            height
        )
    )


    box = (
        x1,
        y1,
        x2,
        y2
    )


    if not is_valid_box(box):
        return None


    mask = np.zeros_like(
        gray_image
    )


    mask[
        y1:y2,
        x1:x2
    ] = 255


    points = cv2.goodFeaturesToTrack(

        gray_image,

        mask=mask,

        maxCorners=30,

        qualityLevel=0.01,

        minDistance=5,

        blockSize=7
    )


    return points


# ==================================================
# SIGLIP VERIFICATION
# ==================================================

def verify_if_needed(
    frame,
    object_name,
    box
):

    if object_name not in CONFUSION_GROUPS:
        return object_name


    x1, y1, x2, y2 = box


    height, width = frame.shape[:2]


    x1 = max(
        0,
        int(x1)
    )

    y1 = max(
        0,
        int(y1)
    )

    x2 = min(
        width,
        int(x2)
    )

    y2 = min(
        height,
        int(y2)
    )


    box = (
        x1,
        y1,
        x2,
        y2
    )


    if not is_valid_box(box):
        return object_name


    box_width = x2 - x1

    box_height = y2 - y1


    pad_x = int(
        box_width * 0.05
    )

    pad_y = int(
        box_height * 0.05
    )


    crop_x1 = max(
        0,
        x1 - pad_x
    )


    crop_y1 = max(
        0,
        y1 - pad_y
    )


    crop_x2 = min(
        width,
        x2 + pad_x
    )


    crop_y2 = min(
        height,
        y2 + pad_y
    )


    crop = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ]


    if crop.size == 0:
        return object_name


    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )


    crop_pil = Image.fromarray(
        crop_rgb
    )


    verification = verify_candidates(

        crop_pil,

        CONFUSION_GROUPS[
            object_name
        ]
    )


    if verification is None:
        return object_name


    return verification[
        "best_label"
    ]


# ==================================================
# BACKGROUND GROUNDING DINO
# ==================================================

def run_detection(
    frame_copy
):

    global detection_running

    global pending_detections


    try:

        raw_detections = detect_objects(

            frame_copy,

            box_threshold=0.20,

            text_threshold=0.20,

            nms_threshold=0.35
        )


        clean_detections = []


        frame_height, frame_width = (
            frame_copy.shape[:2]
        )


        for detection in raw_detections:

            object_name = normalize_live_label(
                detection["object"]
            )


            confidence = detection[
                "confidence"
            ]


            box = tuple(
                map(
                    int,
                    detection["box"]
                )
            )


            # ------------------------------------------
            # Ignore classes outside our useful set
            # ------------------------------------------

            if object_name not in ALLOWED_OBJECTS:
                continue


            # ------------------------------------------
            # Class-specific confidence
            # ------------------------------------------

            if not passes_threshold(
                object_name,
                confidence
            ):

                continue


            # ------------------------------------------
            # Invalid box
            # ------------------------------------------

            if not is_valid_box(
                box,
                frame_width,
                frame_height
            ):

                continue


            # ------------------------------------------
            # Selective SigLIP
            # ------------------------------------------

            final_name = verify_if_needed(

                frame_copy,

                object_name,

                box
            )


            final_name = normalize_live_label(
                final_name
            )


            if final_name not in ALLOWED_OBJECTS:
                continue


            clean_detections.append({

                "object": final_name,

                "confidence": confidence,

                "box": box
            })


        with lock:

            pending_detections = (
                clean_detections
            )


    except Exception as error:

        print(
            "Detection error:",
            error
        )


    finally:

        detection_running = False


# ==================================================
# DETECTION -> TRACK RECONCILIATION
# ==================================================

def update_tracks_with_detections(
    detections,
    gray_image
):

    global tracks
    global next_track_id


    frame_height, frame_width = (
        gray_image.shape[:2]
    )


    frame_diagonal = (
        frame_width * frame_width
        + frame_height * frame_height
    ) ** 0.5


    used_track_ids = set()

    updated_tracks = []


    # ==================================================
    # PROCESS FRESH DETECTIONS
    # ==================================================

    for detection in detections:

        new_box = detection[
            "box"
        ]


        new_label = detection[
            "object"
        ]


        confidence = detection[
            "confidence"
        ]


        if not is_valid_box(
            new_box,
            frame_width,
            frame_height
        ):

            continue


        new_center = get_box_center(
            new_box
        )


        best_track = None

        best_score = 0.0


        # --------------------------------------------------
        # Try to find the same physical object
        # --------------------------------------------------

        for track in tracks:

            if track["id"] in used_track_ids:
                continue


            old_box = track[
                "box"
            ]


            if not is_valid_box(
                old_box,
                frame_width,
                frame_height
            ):

                continue


            iou = calculate_iou(
                new_box,
                old_box
            )


            containment = calculate_containment(
                new_box,
                old_box
            )


            old_center = get_box_center(
                old_box
            )


            dx = (
                new_center[0]
                - old_center[0]
            )


            dy = (
                new_center[1]
                - old_center[1]
            )


            center_distance = (
                dx * dx
                + dy * dy
            ) ** 0.5


            normalized_distance = (
                center_distance
                / frame_diagonal
            )


            # --------------------------------------------------
            # Spatial matching matters more than class name.
            #
            # This allows:
            #
            # desk → bed
            #
            # while keeping the same physical track.
            # --------------------------------------------------

            score = 0.0


            score += (
                iou * 0.60
            )


            score += (
                containment * 0.30
            )


            if normalized_distance < 0.08:

                score += 0.15


            # Label match helps but is NOT required
            if (
                track["object"]
                == new_label
            ):

                score += 0.15


            if score > best_score:

                best_score = score

                best_track = track


        # ==================================================
        # MATCHED EXISTING PHYSICAL OBJECT
        # ==================================================

        if (
            best_track is not None
            and best_score >= 0.30
        ):

            track_id = best_track[
                "id"
            ]


            used_track_ids.add(
                track_id
            )


            confirmations = (
                best_track.get(
                    "confirmations",
                    1
                )
                + 1
            )


            # ----------------------------------------------
            # Label history
            # ----------------------------------------------

            label_votes = best_track.get(
                "label_votes",
                {}
            ).copy()


            label_votes[
                new_label
            ] = (
                label_votes.get(
                    new_label,
                    0
                )
                + 1
            )


            # ----------------------------------------------
            # Majority label wins
            # ----------------------------------------------

            final_label = max(
                label_votes,
                key=label_votes.get
            )


        # ==================================================
        # NEW PHYSICAL OBJECT
        # ==================================================

        else:

            track_id = (
                next_track_id
            )


            next_track_id += 1


            confirmations = 1


            label_votes = {
                new_label: 1
            }


            final_label = (
                new_label
            )


        # --------------------------------------------------
        # Fresh detector box resets optical-flow drift
        # --------------------------------------------------

        points = get_points_inside_box(
            gray_image,
            new_box
        )


        updated_tracks.append({

            "id":
                track_id,

            # Consensus name
            "object":
                final_label,

            # Latest Grounding DINO / SigLIP name
            "latest_label":
                new_label,

            "confidence":
                confidence,

            "box":
                new_box,

            "points":
                points,

            "confirmations":
                confirmations,

            "label_votes":
                label_votes,

            "missed_scans":
                0,

            "fresh_detection":
                True
        })


    # ==================================================
    # TRACKS NOT SEEN THIS SCAN
    # ==================================================

    for old_track in tracks:

        if old_track["id"] in used_track_ids:
            continue


        old_track[
            "missed_scans"
        ] = (
            old_track.get(
                "missed_scans",
                0
            )
            + 1
        )


        old_track[
            "fresh_detection"
        ] = False


        if (
            old_track["missed_scans"]
            <= MAX_MISSED_SCANS

            and is_valid_box(
                old_track["box"],
                frame_width,
                frame_height
            )
        ):

            updated_tracks.append(
                old_track
            )


    tracks = updated_tracks


# ==================================================
# OPTICAL FLOW
# ==================================================

def update_tracks_with_optical_flow(
    previous_gray_frame,
    current_gray_frame
):

    global tracks


    if previous_gray_frame is None:
        return


    frame_height, frame_width = (
        current_gray_frame.shape[:2]
    )


    valid_tracks = []


    for track in tracks:

        # Optical flow is NOT detector confirmation
        track[
            "fresh_detection"
        ] = False


        if not is_valid_box(
            track["box"],
            frame_width,
            frame_height
        ):

            continue


        points = track[
            "points"
        ]


        if (
            points is None
            or len(points) < 3
        ):

            track["points"] = (
                get_points_inside_box(
                    current_gray_frame,
                    track["box"]
                )
            )


            valid_tracks.append(
                track
            )

            continue


        new_points, status, error = (
            cv2.calcOpticalFlowPyrLK(

                previous_gray_frame,

                current_gray_frame,

                points,

                None,

                winSize=(21, 21),

                maxLevel=3,

                criteria=(

                    cv2.TERM_CRITERIA_EPS
                    | cv2.TERM_CRITERIA_COUNT,

                    30,

                    0.01
                )
            )
        )


        if (
            new_points is None
            or status is None
        ):

            valid_tracks.append(
                track
            )

            continue


        valid = (
            status.flatten()
            == 1
        )


        good_old = points[
            valid
        ].reshape(
            -1,
            2
        )


        good_new = new_points[
            valid
        ].reshape(
            -1,
            2
        )


        if len(good_new) < 3:

            track["points"] = (
                get_points_inside_box(
                    current_gray_frame,
                    track["box"]
                )
            )


            valid_tracks.append(
                track
            )

            continue


        movement = (
            good_new
            - good_old
        )


        dx = float(
            np.median(
                movement[:, 0]
            )
        )


        dy = float(
            np.median(
                movement[:, 1]
            )
        )


        x1, y1, x2, y2 = (
            track["box"]
        )


        new_x1 = int(
            x1 + dx
        )

        new_x2 = int(
            x2 + dx
        )

        new_y1 = int(
            y1 + dy
        )

        new_y2 = int(
            y2 + dy
        )


        # ----------------------------------------------
        # Reject completely drifted tracks
        # ----------------------------------------------

        if (
            new_x2 <= 0
            or new_x1 >= frame_width
            or new_y2 <= 0
            or new_y1 >= frame_height
        ):

            continue


        # ----------------------------------------------
        # Clamp only after checking intersection
        # ----------------------------------------------

        new_x1 = max(
            0,
            new_x1
        )

        new_y1 = max(
            0,
            new_y1
        )

        new_x2 = min(
            frame_width,
            new_x2
        )

        new_y2 = min(
            frame_height,
            new_y2
        )


        new_box = (

            new_x1,

            new_y1,

            new_x2,

            new_y2
        )


        # Prevent collapsed boxes
        if not is_valid_box(
            new_box,
            frame_width,
            frame_height
        ):

            continue


        track[
            "box"
        ] = new_box


        track[
            "points"
        ] = good_new.reshape(
            -1,
            1,
            2
        )


        valid_tracks.append(
            track
        )


    tracks = valid_tracks


# ==================================================
# TRUSTED MEMORY GATE
# ==================================================

def get_trusted_tracks():

    trusted_tracks = []


    for track in tracks:

        # Must have enough independent detections
        if (
            track.get(
                "confirmations",
                0
            )
            < MIN_CONFIRMATIONS
        ):

            continue


        # Only fresh detector observations can
        # update memory.
        if not track.get(
            "fresh_detection",
            False
        ):

            continue


        if not is_valid_box(
            track["box"]
        ):

            continue


        trusted_tracks.append(
            track
        )


    return trusted_tracks


# ==================================================
# MAIN VIDEO LOOP
# ==================================================

while True:

    success, frame = video.read()


    if not success:
        break


    frame_number += 1


    current_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------
    # Smooth current tracks
    # --------------------------------------------------

    update_tracks_with_optical_flow(
        previous_gray,
        current_gray
    )


    # --------------------------------------------------
    # Retrieve asynchronous detector result
    # --------------------------------------------------

    new_detections = None


    with lock:

        if pending_detections is not None:

            new_detections = (
                pending_detections
            )

            pending_detections = None


    detector_updated = False


    # --------------------------------------------------
    # Reconcile fresh detections with tracks
    # --------------------------------------------------

    if new_detections is not None:

        update_tracks_with_detections(

            new_detections,

            current_gray
        )


        detector_updated = True


    # --------------------------------------------------
    # TRUSTED MEMORY UPDATE
    # --------------------------------------------------

    if detector_updated:

        trusted_tracks = (
            get_trusted_tracks()
        )


        if trusted_tracks:

            update_memory(

                trusted_tracks,

                frame_number
            )


    # --------------------------------------------------
    # Start next asynchronous scan
    # --------------------------------------------------

    if (
        frame_number
        % PROCESS_EVERY
        == 0

        and not detection_running
    ):

        detection_running = True


        detection_thread = (
            threading.Thread(

                target=run_detection,

                args=(
                    frame.copy(),
                ),

                daemon=True
            )
        )


        detection_thread.start()


    # ==================================================
    # DRAW TRACKS
    # ==================================================

    for track in tracks:

        if not is_valid_box(
            track["box"]
        ):

            continue


        object_name = track[
            "object"
        ]


        latest_label = track.get(
            "latest_label",
            object_name
        )


        confidence = track[
            "confidence"
        ]


        track_id = track[
            "id"
        ]


        confirmations = track.get(
            "confirmations",
            0
        )


        x1, y1, x2, y2 = (
            track["box"]
        )


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2
        )


        label = (

            f"{object_name} "

            f"| ID:{track_id} "

            f"| {confidence * 100:.0f}% "

            f"| C:{confirmations}"
        )


        cv2.putText(

            frame,

            label,

            (
                x1,
                max(
                    20,
                    y1 - 8
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (0, 255, 0),

            1,

            cv2.LINE_AA
        )


        # --------------------------------------------------
        # If detector currently disagrees with consensus,
        # display that too for debugging.
        # --------------------------------------------------

        if (
            latest_label
            != object_name
        ):

            cv2.putText(

                frame,

                f"latest: {latest_label}",

                (
                    x1,
                    min(
                        frame.shape[0] - 10,
                        y2 + 18
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.40,

                (0, 255, 255),

                1,

                cv2.LINE_AA
            )


    # --------------------------------------------------
    # Memory counter
    # --------------------------------------------------

    cv2.putText(

        frame,

        f"Memory: {len(get_memory())}",

        (
            20,
            65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (0, 255, 255),

        2
    )


    # --------------------------------------------------
    # Scanning indicator
    # --------------------------------------------------

    if detection_running:

        cv2.putText(

            frame,

            "Scanning...",

            (
                20,
                35
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 255),

            2
        )


    cv2.imshow(

        "BHASKARA - Reconciled Tracking",

        frame
    )


    previous_gray = (
        current_gray.copy()
    )


    key = cv2.waitKey(
        delay
    ) & 0xFF


    if key == ord("q"):
        break


# ==================================================
# CLEANUP
# ==================================================

video.release()

cv2.destroyAllWindows()


# ==================================================
# FINAL MEMORY REPORT
# ==================================================

print("\n")

print(
    "=" * 70
)

print(
    "BHASKARA TRUSTED OBJECT MEMORY"
)

print(
    "=" * 70
)


memory = get_memory()


if not memory:

    print(
        "No objects passed the trusted-memory gate."
    )


else:

    for track_id, data in memory.items():

        print(
            f"ID:{track_id}"
        )

        print(
            f"  Object: {data['object']}"
        )

        print(
            f"  Confidence: "
            f"{data['confidence'] * 100:.2f}%"
        )

        print(
            f"  Confirmations: "
            f"{data['confirmations']}"
        )

        print(
            f"  Label votes: "
            f"{data.get('label_votes', {})}"
        )

        print(
            f"  Last frame: "
            f"{data['last_seen_frame']}"
        )

        print(
            f"  Time: "
            f"{data['last_seen_time'].strftime('%H:%M:%S')}"
        )

        print(
            f"  Box: "
            f"{data['box']}"
        )

        print(
            "-" * 40
        )


print(
    "=" * 70
)


print(
    "BHASKARA reconciliation test finished."
)