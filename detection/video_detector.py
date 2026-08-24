# --------------------------------------------------
# BHASKARA
# Asynchronous Video Detection
#
# Main thread:
#   reads and displays video smoothly
#
# Background thread:
#   runs Grounding DINO periodically
# --------------------------------------------------

import cv2
import threading

from detection.grounding_detector import detect_objects


# --------------------------------------------------
# 1. Video path
# --------------------------------------------------

VIDEO_PATH = "videos/room.mp4"


# --------------------------------------------------
# 2. Open video
# --------------------------------------------------

video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Could not open video.")
    exit()


# --------------------------------------------------
# 3. Get original video FPS
# --------------------------------------------------

fps = video.get(
    cv2.CAP_PROP_FPS
)

# Fallback in case FPS cannot be read
if fps <= 0:
    fps = 30


# Calculate how long each frame should stay visible
delay = int(
    1000 / fps
)

print(
    "Video FPS:",
    round(fps, 2)
)


# --------------------------------------------------
# 4. Processing settings
# --------------------------------------------------

frame_number = 0

# Run Grounding DINO every few frames
PROCESS_EVERY = 5


# --------------------------------------------------
# 5. Shared detection state
# --------------------------------------------------

# Most recent detections produced
# by the background detector
last_detections = []

# True when the detector is currently working
detection_running = False

# Prevent two threads from changing shared
# detection data at the same time
lock = threading.Lock()


# --------------------------------------------------
# 6. Background detection function
# --------------------------------------------------

def run_detection(frame_copy):

    global last_detections
    global detection_running


    try:

        # ----------------------------------------------
        # Run Grounding DINO
        # ----------------------------------------------

        detections = detect_objects(
            frame_copy,
            box_threshold=0.25,
            text_threshold=0.25,
            nms_threshold=0.35
        )


        # ----------------------------------------------
        # Safely update latest detections
        # ----------------------------------------------

        with lock:

            last_detections = detections


    except Exception as error:

        print(
            "Detection error:",
            error
        )


    finally:

        # Detection finished
        detection_running = False


# --------------------------------------------------
# 7. Main video loop
# --------------------------------------------------

while True:

    success, frame = video.read()

    if not success:
        break


    frame_number += 1


    # --------------------------------------------------
    # 8. Start detection periodically
    # --------------------------------------------------

    if (
        frame_number % PROCESS_EVERY == 0
        and not detection_running
    ):

        detection_running = True


        # Give background detector its own frame copy
        frame_copy = frame.copy()


        detection_thread = threading.Thread(
            target=run_detection,
            args=(frame_copy,),
            daemon=True
        )


        detection_thread.start()


    # --------------------------------------------------
    # 9. Copy latest detections safely
    # --------------------------------------------------

    with lock:

        detections_to_draw = list(
            last_detections
        )


    # --------------------------------------------------
    # 10. Draw latest detections
    # --------------------------------------------------

    for detection in detections_to_draw:

        object_name = detection[
            "object"
        ]

        confidence = detection[
            "confidence"
        ]

        x1, y1, x2, y2 = detection[
            "box"
        ]


        # ----------------------------------------------
        # Ignore weak detections
        # ----------------------------------------------

        if confidence < 0.30:
            continue


        # ----------------------------------------------
        # Draw bounding box
        # ----------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------
        # Create label
        # ----------------------------------------------

        label = (
            f"{object_name} "
            f"{confidence * 100:.0f}%"
        )


        # ----------------------------------------------
        # Calculate text size
        # ----------------------------------------------

        (
            text_width,
            text_height
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )


        label_y = max(
            y1,
            text_height + 10
        )


        # ----------------------------------------------
        # Draw filled label background
        # ----------------------------------------------

        cv2.rectangle(
            frame,
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


        # ----------------------------------------------
        # Draw label text
        # ----------------------------------------------

        cv2.putText(
            frame,
            label,
            (
                x1 + 4,
                label_y - 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )


    # --------------------------------------------------
    # 11. Show scanning status
    # --------------------------------------------------

    if detection_running:

        cv2.putText(
            frame,
            "Scanning...",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


    # --------------------------------------------------
    # 12. Display frame
    # --------------------------------------------------

    cv2.imshow(
        "BHASKARA - Async Grounding DINO",
        frame
    )


    # --------------------------------------------------
    # 13. Maintain original video speed
    # --------------------------------------------------

    key = cv2.waitKey(
        delay
    ) & 0xFF


    # Press Q to stop
    if key == ord("q"):
        break


# --------------------------------------------------
# 14. Cleanup
# --------------------------------------------------

video.release()

cv2.destroyAllWindows()

print(
    "Video detection finished."
)