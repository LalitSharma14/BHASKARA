# --------------------------------------------------
# BHASKARA
# YOLO-World + SigLIP Verification Test
# --------------------------------------------------

import cv2
from PIL import Image

from detection.detector import detect_objects
from verification.verifier import (
    verify_detection,
    CONFUSION_GROUPS
)


# --------------------------------------------------
# 1. Image to test
# --------------------------------------------------

IMAGE_PATH = "images/room.jpeg"


# --------------------------------------------------
# 2. Load original image
# --------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Could not load image.")
    exit()


print("\nBHASKARA TWO-STAGE VISION TEST")
print("-----------------------------------")


# --------------------------------------------------
# 3. Run YOLO-World detection
# --------------------------------------------------

detections = detect_objects(IMAGE_PATH)

print(
    "YOLO detections:",
    len(detections)
)

print()


# --------------------------------------------------
# 4. Process each YOLO detection
# --------------------------------------------------

for detection in detections:

    yolo_name = detection["object"]

    confidence = detection["confidence"]

    x1, y1, x2, y2 = detection["box"]


    print("-----------------------------------")

    print(
        "YOLO prediction:",
        yolo_name
    )

    print(
        "YOLO confidence:",
        round(confidence * 100, 2),
        "%"
    )


    # --------------------------------------------------
    # 5. Default final label = YOLO label
    # --------------------------------------------------

    final_name = yolo_name


    # --------------------------------------------------
    # 6. Check whether this class needs verification
    # --------------------------------------------------

    if yolo_name in CONFUSION_GROUPS:

        print("Ambiguous class -> running SigLIP...")


        # --------------------------------------------------
        # 7. Add a SMALL amount of padding
        # --------------------------------------------------
        # YOLO boxes can sometimes cut off important parts
        # of the object.
        #
        # We expand the box slightly before verification.
        # --------------------------------------------------

        height, width = image.shape[:2]

        box_width = x2 - x1
        box_height = y2 - y1


        # 5% padding
        pad_x = int(box_width * 0.05)
        pad_y = int(box_height * 0.05)


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


        # --------------------------------------------------
        # 8. Automatically crop YOLO region
        # --------------------------------------------------

        crop = image[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]


        if crop.size == 0:

            print("Invalid crop. Keeping YOLO prediction.")

        else:

            # --------------------------------------------------
            # 9. Convert OpenCV BGR -> RGB
            # --------------------------------------------------

            crop_rgb = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )


            # Convert into PIL image for SigLIP
            crop_pil = Image.fromarray(
                crop_rgb
            )


            # --------------------------------------------------
            # 10. Run SigLIP verification
            # --------------------------------------------------

            verification = verify_detection(
                crop_pil,
                yolo_name
            )


            if verification is not None:

                print("\nSigLIP scores:")


                for label, score in verification["scores"].items():

                    print(
                        " ",
                        label,
                        ":",
                        round(score * 100, 2),
                        "%"
                    )


                # --------------------------------------------------
                # 11. Get SigLIP's best description
                # --------------------------------------------------

                best_description = verification[
                    "best_label"
                ]

                best_score = verification[
                    "best_score"
                ]


                print(
                    "\nSigLIP best:",
                    best_description
                )

                print(
                    "Relative score:",
                    round(best_score * 100, 2),
                    "%"
                )


                # --------------------------------------------------
                # 12. Convert long SigLIP description
                #     into simple BHASKARA label
                # --------------------------------------------------

                description_lower = (
                    best_description.lower()
                )


                if (
                    "earphone" in description_lower
                    or "earbud" in description_lower
                ):

                    final_name = "wired earphones"


                elif (
                    "charging cable" in description_lower
                    or "phone charger" in description_lower
                ):

                    final_name = "charger"


                elif "usb cable" in description_lower:

                    final_name = "USB cable"


                elif "window" in description_lower:

                    final_name = "window"


                elif "cabinet" in description_lower:

                    final_name = "cabinet"


                elif "door" in description_lower:

                    final_name = "door"


    else:

        print(
            "Not ambiguous -> SigLIP skipped."
        )


    # --------------------------------------------------
    # 13. Print final decision
    # --------------------------------------------------

    print(
        "\nFINAL:",
        final_name
    )


    # --------------------------------------------------
    # 14. Draw bounding box
    # --------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # --------------------------------------------------
    # 15. Draw final label
    # --------------------------------------------------

    label = (
        f"{final_name} "
        f"{confidence * 100:.0f}%"
    )


    cv2.putText(
        image,
        label,
        (
            x1,
            max(y1 - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# --------------------------------------------------
# 16. Show final result
# --------------------------------------------------

cv2.imshow(
    "BHASKARA - YOLO + SigLIP",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()


print("\nVerification test finished.")