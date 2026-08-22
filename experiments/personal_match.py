import cv2
import numpy as np


# Load reference image
reference = cv2.imread(
    "reference_object/wired_earphones/wired_earphones_1.jpeg"
)


# Load test image
test_image = cv2.imread(
    "images/wired_earphones_test.jpeg"
)


# Check whether images loaded properly
if reference is None:
    print("Could not load reference image")
    exit()

if test_image is None:
    print("Could not load test image")
    exit()


# Convert both images to grayscale
reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY
)

test_gray = cv2.cvtColor(
    test_image,
    cv2.COLOR_BGR2GRAY
)


# Create SIFT detector
sift = cv2.SIFT_create()


# Detect keypoints and descriptors
keypoints_ref, descriptors_ref = sift.detectAndCompute(
    reference_gray,
    None
)

keypoints_test, descriptors_test = sift.detectAndCompute(
    test_gray,
    None
)


# Print number of detected features
print(
    "Reference features:",
    len(keypoints_ref)
)

print(
    "Test image features:",
    len(keypoints_test)
)


# Create brute-force matcher
matcher = cv2.BFMatcher()


# Find two nearest matches for each feature
matches = matcher.knnMatch(
    descriptors_ref,
    descriptors_test,
    k=2
)


# Lowe's Ratio Test
good_matches = []

for m, n in matches:

    if m.distance < 0.65 * n.distance:
        good_matches.append(m)


print(
    "Good matches:",
    len(good_matches)
)


# Draw some matches for visualization
matched_image = cv2.drawMatches(
    reference,
    keypoints_ref,
    test_image,
    keypoints_test,
    good_matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


cv2.imshow(
    "BHASKARA - Feature Matches",
    matched_image
)


# Continue only if enough good matches exist
if len(good_matches) >= 10:

    # Get coordinates of matched points
    src_points = np.float32(
        [
            keypoints_ref[m.queryIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)


    dst_points = np.float32(
        [
            keypoints_test[m.trainIdx].pt
            for m in good_matches
        ]
    ).reshape(-1, 1, 2)


    # Calculate homography using RANSAC
    matrix, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        5.0
    )

    print("Homography matrix:")
    print(matrix)

    if mask is not None:
        print("Inlier matches:", int(mask.sum()))

    # Check whether homography was successfully calculated
    if matrix is not None:

        # Get reference image dimensions
        h, w = reference_gray.shape


        # Four corners of reference image
        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)


        # Project the reference corners onto room image
        projected_corners = cv2.perspectiveTransform(
            corners,
            matrix
        )


        # Convert points into integers
        projected_corners = np.int32(
            projected_corners
        )


        # Draw detected area
        detected_image = cv2.polylines(
            test_image.copy(),
            [projected_corners],
            True,
            (0, 255, 0),
            4
        )


        # Add label
        first_point = projected_corners[0][0]

        x = first_point[0]
        y = first_point[1]


        cv2.putText(
            detected_image,
            "Wired Earphones",
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # Show detection result
        cv2.imshow(
            "BHASKARA - Earphones Located",
            detected_image
        )


    else:

        print(
            "Homography could not be calculated."
        )


else:

    print(
        "Not enough good matches to locate earphones."
    )


# Wait until key is pressed
cv2.waitKey(0)


# Close all windows
cv2.destroyAllWindows()