import cv2

from detection.final_detector import detect_final_objects


VIDEO_PATH = "videos/room.mp4"


# Open video
video = cv2.VideoCapture(VIDEO_PATH)

if not video.isOpened():
    print("Could not open video.")
    exit()


# Read only the first frame
success, frame = video.read()

video.release()


if not success:
    print("Could not read first video frame.")
    exit()


print("\nRunning BHASKARA on one video frame...\n")


# Send OpenCV frame directly to final detector
detections = detect_final_objects(
    frame
)


print(
    "Objects detected:",
    len(detections)
)

print()


for detection in detections:

    print(
        detection["object"],
        "-",
        round(
            detection["confidence"] * 100,
            2
        ),
        "%",
        "|",
        detection["type"],
        "| Box:",
        detection["box"]
    )


print(
    "\nSingle-frame test finished."
)