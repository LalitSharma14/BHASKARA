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
from verification.verifier import (
    get_image_embeddings,
    verify_candidates
)

from memory.object_memory import (
    update_memory,
    get_memory,
    get_memory_stats,
    get_audit_run_directory,
    write_audit_summary
)

from tracking.reconciliation import (
    find_global_assignments,
    find_lost_track_assignments,
    select_consensus_label,
    update_appearance_gallery,
    update_appearance_prototype
)
from tracking.appearance_quality import describe_crop
from tracking.detection_timing import (
    calculate_result_age,
    select_reconciled_box,
    should_accept_result
)
from tracking.motion_compensation import transport_detections


# ==================================================
# CONFIGURATION
# ==================================================

VIDEO_PATH = "videos/room.mp4"


# Run Grounding DINO periodically
PROCESS_EVERY = 5


# Object must be independently detected this
# many times before entering memory
MIN_CONFIRMATIONS = 3


MEMORY_CONFIRMATION_REQUIREMENTS = {
    "id card": 7,
    "usb cable": 6,
    "wireless earbuds": 6,
    "wired earphones": 5,
    "charger": 5,
    "clothes": 5,
    "metal ruler": 5,
    "keys": 5
}


MIN_MEMORY_HIT_RATIO = 0.55
MIN_MEMORY_LABEL_STABILITY = 0.70


# Keep missed tracks alive briefly
MAX_MISSED_SCANS = 5


# Recently expired tracks remain eligible for
# conservative exact-label recovery.
MAX_LOST_TRACK_AGE_FRAMES = 180


# Ignore detector results that describe a frame
# too far behind the current video position.
MAX_DETECTION_RESULT_AGE = 45


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

last_detection_age = None

last_motion_compensation = None


lock = threading.Lock()


tracks = []

lost_tracks = []

next_track_id = 1


runtime_stats = {
    "detector_jobs_started": 0,
    "detector_results_accepted": 0,
    "detector_results_stale": 0,
    "detections_received": 0,
    "motion_aligned": 0,
    "appearance_embeddings_received": 0,
    "active_matches": 0,
    "lost_tracks_recovered": 0,
    "reid_same_label_comparisons": 0,
    "reid_above_threshold": 0,
    "reid_ambiguous": 0,
    "reid_similarity_sum": 0.0,
    "reid_similarity_count": 0,
    "reid_similarity_max": 0.0,
    "new_tracks_created": 0,
    "tracks_expired": 0
}


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
# BATCHED APPEARANCE IDENTITY
# ==================================================

def attach_appearance_embeddings(frame, detections, source_frame_number):
    """Attach one object-focused embedding to each reliable detection crop.

    Context fields remain optional for schema compatibility, but context crops
    are not embedded because they doubled SigLIP work and reduced detector
    throughput without providing enough identity value.
    """

    frame_height, frame_width = frame.shape[:2]
    crops = []
    embedding_targets = []

    for index, detection in enumerate(detections):
        description = describe_crop(
            detection["box"],
            frame_width,
            frame_height,
        )
        x1, y1, x2, y2 = description["box"]

        detection["appearance_quality"] = description["quality"]
        detection["appearance_aspect_ratio"] = description["aspect_ratio"]
        detection["appearance_tiny"] = description["tiny"]
        detection["appearance_touches_edge"] = description["touches_edge"]
        detection["context_embedding"] = None

        if not description["valid"]:
            detection["appearance_embedding"] = None
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            detection["appearance_embedding"] = None
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(Image.fromarray(crop_rgb))
        embedding_targets.append((index, "appearance_embedding"))

        detection["evidence_crop"] = crop.copy()
        detection["evidence_frame"] = source_frame_number
        detection["evidence_confidence"] = detection["confidence"]

    if not crops:
        return detections

    try:
        embeddings = get_image_embeddings(crops)

        for (detection_index, embedding_key), embedding in zip(
            embedding_targets,
            embeddings
        ):
            detections[detection_index][embedding_key] = embedding

    except Exception as error:
        print("Appearance embedding error:", error)

        for detection_index, _embedding_key in embedding_targets:
            detections[detection_index]["appearance_embedding"] = None
            detections[detection_index]["context_embedding"] = None

    return detections


# ==================================================
# BACKGROUND GROUNDING DINO
# ==================================================

def run_detection(
    frame_copy,
    source_frame_number
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


        clean_detections = attach_appearance_embeddings(
            frame_copy,
            clean_detections,
            source_frame_number
        )


        with lock:

            pending_detections = {
                "detections": clean_detections,
                "source_frame_number": source_frame_number,
                "source_gray": cv2.cvtColor(
                    frame_copy,
                    cv2.COLOR_BGR2GRAY
                )
            }


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
    gray_image,
    detection_age_frames=0,
    current_frame_number=0
):

    global tracks
    global lost_tracks
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


    lost_tracks = [
        track
        for track in lost_tracks
        if (
            current_frame_number
            - track.get("lost_at_frame", current_frame_number)
            <= MAX_LOST_TRACK_AGE_FRAMES
        )
    ]


    assignments = find_global_assignments(

        detections,

        tracks,

        frame_diagonal,

        lambda box: is_valid_box(
            box,
            frame_width,
            frame_height
        )
    )


    unmatched_indexes = [
        index
        for index in range(len(detections))
        if index not in assignments
    ]


    unmatched_detections = [
        detections[index]
        for index in unmatched_indexes
    ]


    (
        local_lost_assignments,
        reid_diagnostics
    ) = find_lost_track_assignments(

        unmatched_detections,

        lost_tracks,

        frame_diagonal,

        lambda box: is_valid_box(
            box,
            frame_width,
            frame_height
        ),

        return_diagnostics=True
    )


    runtime_stats["reid_same_label_comparisons"] += (
        reid_diagnostics["same_label_comparisons"]
    )
    runtime_stats["reid_above_threshold"] += (
        reid_diagnostics["above_threshold"]
    )
    runtime_stats["reid_ambiguous"] += reid_diagnostics["ambiguous"]


    for similarity in reid_diagnostics["best_similarities"]:
        runtime_stats["reid_similarity_sum"] += similarity
        runtime_stats["reid_similarity_count"] += 1
        runtime_stats["reid_similarity_max"] = max(
            runtime_stats["reid_similarity_max"],
            similarity
        )


    lost_assignments = {
        unmatched_indexes[local_index]: assignment
        for local_index, assignment in local_lost_assignments.items()
    }


    recovered_track_ids = set()


    # ==================================================
    # PROCESS FRESH DETECTIONS
    # ==================================================

    for detection_index, detection in enumerate(detections):

        new_box = detection[
            "box"
        ]


        new_label = detection[
            "object"
        ]


        confidence = detection[
            "confidence"
        ]


        appearance_observation = detection.get(
            "appearance_embedding"
        )

        context_observation = detection.get("context_embedding")


        if not is_valid_box(
            new_box,
            frame_width,
            frame_height
        ):

            continue


        best_track, best_score = assignments.get(
            detection_index,
            lost_assignments.get(
                detection_index,
                (None, 0.0)
            )
        )


        recovered_from_lost = (
            detection_index in lost_assignments
            and detection_index not in assignments
        )


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


            if recovered_from_lost:

                recovered_track_ids.add(track_id)
                runtime_stats["lost_tracks_recovered"] += 1


            else:

                runtime_stats["active_matches"] += 1


            confirmations = (
                best_track.get(
                    "confirmations",
                    1
                )
                + 1
            )


            detector_observations = (
                best_track.get("detector_observations", 1)
                + 1
            )


            appearance_embedding = update_appearance_prototype(
                best_track.get("appearance_embedding"),
                appearance_observation
            )


            appearance_gallery = update_appearance_gallery(
                best_track.get("appearance_gallery", []),
                appearance_observation
            )

            context_embedding = update_appearance_prototype(
                best_track.get("context_embedding"),
                context_observation
            )

            context_gallery = update_appearance_gallery(
                best_track.get("context_gallery", []),
                context_observation
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

            final_label = select_consensus_label(
                label_votes,
                new_label
            )


        # ==================================================
        # NEW PHYSICAL OBJECT
        # ==================================================

        else:

            track_id = (
                next_track_id
            )


            next_track_id += 1


            runtime_stats["new_tracks_created"] += 1


            confirmations = 1


            detector_observations = 1


            appearance_embedding = update_appearance_prototype(
                None,
                appearance_observation
            )


            appearance_gallery = update_appearance_gallery(
                [],
                appearance_observation
            )

            context_embedding = update_appearance_prototype(
                None,
                context_observation
            )

            context_gallery = update_appearance_gallery(
                [],
                context_observation
            )


            label_votes = {
                new_label: 1
            }


            final_label = (
                new_label
            )


        # --------------------------------------------------
        # Fresh detector box resets optical-flow drift
        # --------------------------------------------------

        reconciled_box = new_box


        if (
            best_track is not None
            and best_score >= 0.30
            and not recovered_from_lost
        ):

            reconciled_box = select_reconciled_box(
                new_box,
                best_track["box"],
                detection_age_frames
            )


        points = get_points_inside_box(
            gray_image,
            reconciled_box
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
                reconciled_box,

            "points":
                points,

            "confirmations":
                confirmations,

            "detector_observations":
                detector_observations,

            "label_votes":
                label_votes,

            "appearance_embedding":
                appearance_embedding,

            "appearance_gallery":
                appearance_gallery,

            "context_embedding":
                context_embedding,

            "context_gallery":
                context_gallery,

            "appearance_quality":
                detection.get("appearance_quality", 0.0),

            "appearance_aspect_ratio":
                detection.get("appearance_aspect_ratio"),

            "appearance_tiny":
                detection.get("appearance_tiny", False),

            "appearance_touches_edge":
                detection.get("appearance_touches_edge", False),

            "evidence_crop":
                detection.get("evidence_crop"),

            "evidence_frame":
                detection.get("evidence_frame", current_frame_number),

            "evidence_confidence":
                detection.get("evidence_confidence", confidence),

            "missed_scans":
                0,

            "fresh_detection":
                True
        })


    if recovered_track_ids:

        lost_tracks = [
            track
            for track in lost_tracks
            if track["id"] not in recovered_track_ids
        ]


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


        old_track["detector_observations"] = (
            old_track.get("detector_observations", 1)
            + 1
        )


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


        else:

            lost_track = old_track.copy()
            lost_track["lost_at_frame"] = current_frame_number
            lost_tracks.append(lost_track)
            runtime_stats["tracks_expired"] += 1


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

        required_confirmations = MEMORY_CONFIRMATION_REQUIREMENTS.get(
            track["object"],
            MIN_CONFIRMATIONS
        )

        # Must have enough independent detections
        if (
            track.get(
                "confirmations",
                0
            )
            < required_confirmations
        ):

            continue


        detector_observations = max(
            1,
            track.get("detector_observations", 1)
        )


        hit_ratio = (
            track.get("confirmations", 0)
            / detector_observations
        )


        if hit_ratio < MIN_MEMORY_HIT_RATIO:
            continue


        winning_votes = max(
            track.get("label_votes", {track["object"]: 0}).values()
        )


        label_stability = (
            winning_votes
            / max(1, track.get("confirmations", 0))
        )


        if label_stability < MIN_MEMORY_LABEL_STABILITY:
            continue


        if track.get("appearance_embedding") is None:
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

    detection_result = None


    with lock:

        if pending_detections is not None:

            detection_result = (
                pending_detections
            )

            pending_detections = None


    detector_updated = False


    # --------------------------------------------------
    # Reconcile fresh detections with tracks
    # --------------------------------------------------

    if detection_result is not None:

        result_age = calculate_result_age(
            frame_number,
            detection_result["source_frame_number"]
        )


        last_detection_age = result_age


        if should_accept_result(
            result_age,
            MAX_DETECTION_RESULT_AGE
        ):

            runtime_stats["detector_results_accepted"] += 1

            detections_to_reconcile = detection_result[
                "detections"
            ]


            if result_age > 0:

                (
                    detections_to_reconcile,
                    last_motion_compensation
                ) = transport_detections(

                    detection_result["source_gray"],

                    current_gray,

                    detections_to_reconcile
                )


                runtime_stats["motion_aligned"] += (
                    last_motion_compensation["successful"]
                )


            else:

                last_motion_compensation = {
                    "successful": 0,
                    "total": len(detections_to_reconcile)
                }

            update_tracks_with_detections(

                detections_to_reconcile,

                current_gray,

                detection_age_frames=result_age,

                current_frame_number=frame_number
            )


            runtime_stats["detections_received"] += len(
                detections_to_reconcile
            )


            runtime_stats["appearance_embeddings_received"] += sum(
                detection.get("appearance_embedding") is not None
                for detection in detections_to_reconcile
            )


            detector_updated = True


        else:

            runtime_stats["detector_results_stale"] += 1

            print(
                "Discarding stale detection result:",
                result_age,
                "frames old"
            )


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


        runtime_stats["detector_jobs_started"] += 1


        detection_thread = (
            threading.Thread(

                target=run_detection,

                args=(
                    frame.copy(),
                    frame_number
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
    # Last asynchronous result age
    # --------------------------------------------------

    if last_detection_age is not None:

        cv2.putText(

            frame,

            f"Detector age: {last_detection_age} frames",

            (
                20,
                95
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 255),

            1
        )


    if last_motion_compensation is not None:

        cv2.putText(

            frame,

            (
                "Motion aligned: "
                f"{last_motion_compensation['successful']}/"
                f"{last_motion_compensation['total']}"
            ),

            (
                20,
                120
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 255),

            1
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

    for memory_id, data in memory.items():

        print(
            f"Memory ID:{memory_id}"
        )

        print(
            f"  Track IDs: {data.get('track_ids', [])}"
        )

        print(
            f"  Appearance views: "
            f"{len(data.get('appearance_gallery', []))}"
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
    "BHASKARA TRACKING DIAGNOSTICS"
)


print(
    "=" * 70
)


for statistic_name, statistic_value in runtime_stats.items():

    print(
        f"{statistic_name}: {statistic_value}"
    )


for statistic_name, statistic_value in get_memory_stats().items():

    print(
        f"memory_{statistic_name}: {statistic_value}"
    )


print(
    f"active_tracks_at_end: {len(tracks)}"
)


print(
    f"lost_tracks_at_end: {len(lost_tracks)}"
)


print(
    f"memory_entries_at_end: {len(memory)}"
)


audit_diagnostics = runtime_stats.copy()


for statistic_name, statistic_value in get_memory_stats().items():
    audit_diagnostics[f"memory_{statistic_name}"] = statistic_value


audit_diagnostics["active_tracks_at_end"] = len(tracks)
audit_diagnostics["lost_tracks_at_end"] = len(lost_tracks)
audit_diagnostics["memory_entries_at_end"] = len(memory)


audit_summary_path = write_audit_summary(
    audit_diagnostics
)


print(
    f"identity_audit: {get_audit_run_directory()}"
)


print(
    f"identity_audit_summary: {audit_summary_path}"
)


print(
    "=" * 70
)


print(
    "BHASKARA reconciliation test finished."
)
