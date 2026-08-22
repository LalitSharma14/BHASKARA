# --------------------------------------------------
# BHASKARA
# Main Application
# --------------------------------------------------

# Import our object detection function
# from detection/detector.py
from detection.detector import detect_objects


# --------------------------------------------------
# 1. START BHASKARA
# --------------------------------------------------

print("\nBHASKARA starting...\n")


# --------------------------------------------------
# 2. IMAGE TO ANALYZE
# --------------------------------------------------

image_path = "images/room.jpeg"


# --------------------------------------------------
# 3. RUN OBJECT DETECTION
# --------------------------------------------------

print("Scanning image...\n")

detections = detect_objects(image_path)


# --------------------------------------------------
# 4. PRINT NUMBER OF OBJECTS FOUND
# --------------------------------------------------

print(
    "Objects detected:",
    len(detections)
)

print()


# --------------------------------------------------
# 5. DISPLAY DETECTION INFORMATION
# --------------------------------------------------

for detection in detections:

    # Get information from detector.py
    object_name = detection["object"]
    confidence = detection["confidence"]
    box = detection["box"]

    # Convert confidence from 0-1 to percentage
    confidence_percentage = confidence * 100

    # Print object information
    print(
        object_name,
        "-",
        round(confidence_percentage, 2),
        "%",
        "| Box:",
        box
    )


# --------------------------------------------------
# 6. FINISHED
# --------------------------------------------------

print("\nBHASKARA scan complete.")