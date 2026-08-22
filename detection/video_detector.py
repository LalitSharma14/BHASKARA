import cv2

from tracking.tracker import track_objects


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
# 3. Frame skipping
# --------------------------------------------------

frame_number = 0

# Process every second frame for speed
PROCESS_EVERY = 2

# Store latest tracking results
last_tracked_objects = []


# --------------------------------------------------
# 4. Process video frame by frame
# --------------------------------------------------

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1


    # --------------------------------------------------
    # Run tracking only on selected frames
    # --------------------------------------------------

    if frame_number % PROCESS_EVERY == 0:

        last_tracked_objects = track_objects(
            frame
        )


    # --------------------------------------------------
    # Draw tracked objects
    # --------------------------------------------------

    for tracked_object in last_tracked_objects:

        object_name = tracked_object["object"]

        confidence = tracked_object["confidence"]

        track_id = tracked_object["track_id"]

        x1, y1, x2, y2 = tracked_object["box"]


        # --------------------------------------------------
        # Draw bounding box
        # --------------------------------------------------

        cv2.rectangle(
            frame,
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
            f"| ID:{track_id} "
            f"| {confidence * 100:.0f}%"
        )


        # --------------------------------------------------
        # Draw label background
        # --------------------------------------------------

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        label_y = max(
            y1,
            text_height + 10
        )

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


        # --------------------------------------------------
        # Draw label text
        # --------------------------------------------------

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
    # Display result
    # --------------------------------------------------

    cv2.imshow(
        "BHASKARA - Tracking",
        frame
    )


    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 5. Cleanup
# --------------------------------------------------

video.release()

cv2.destroyAllWindows()

print("Tracking finished.")