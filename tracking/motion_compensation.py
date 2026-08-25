"""Motion compensation for delayed asynchronous detections."""

import cv2
import numpy as np


def transport_box_with_optical_flow(source_gray, current_gray, box):
    """Translate a source-frame box into current-frame coordinates.

    Returns ``(transported_box, diagnostics)``. If reliable motion cannot be
    estimated, the original box is returned with ``success`` set to ``False``.
    """

    diagnostics = {
        "success": False,
        "tracked_points": 0,
        "dx": 0.0,
        "dy": 0.0,
    }

    if source_gray is None or current_gray is None:
        return box, diagnostics

    if source_gray.shape[:2] != current_gray.shape[:2]:
        return box, diagnostics

    height, width = source_gray.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return box, diagnostics

    mask = np.zeros_like(source_gray)
    mask[y1:y2, x1:x2] = 255

    source_points = cv2.goodFeaturesToTrack(
        source_gray,
        mask=mask,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
    )

    if source_points is None or len(source_points) < 3:
        return box, diagnostics

    current_points, status, errors = cv2.calcOpticalFlowPyrLK(
        source_gray,
        current_gray,
        source_points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            40,
            0.01,
        ),
    )

    if current_points is None or status is None:
        return box, diagnostics

    valid = status.flatten() == 1
    if errors is not None:
        valid &= np.isfinite(errors.flatten())

    old_points = source_points[valid].reshape(-1, 2)
    new_points = current_points[valid].reshape(-1, 2)

    if len(new_points) < 3:
        return box, diagnostics

    movement = new_points - old_points
    median_movement = np.median(movement, axis=0)
    deviations = np.linalg.norm(movement - median_movement, axis=1)
    median_deviation = float(np.median(deviations))
    tolerance = max(2.0, median_deviation * 3.0)
    inliers = deviations <= tolerance

    if int(inliers.sum()) < 3:
        return box, diagnostics

    dx, dy = np.median(movement[inliers], axis=0)
    dx = float(dx)
    dy = float(dy)

    transported_box = (
        max(0, min(width, int(round(x1 + dx)))),
        max(0, min(height, int(round(y1 + dy)))),
        max(0, min(width, int(round(x2 + dx)))),
        max(0, min(height, int(round(y2 + dy)))),
    )

    if (
        transported_box[2] - transported_box[0] < 10
        or transported_box[3] - transported_box[1] < 10
    ):
        return box, diagnostics

    diagnostics.update(
        success=True,
        tracked_points=int(inliers.sum()),
        dx=dx,
        dy=dy,
    )
    return transported_box, diagnostics


def transport_detections(source_gray, current_gray, detections):
    """Motion-compensate a detection list and return aggregate diagnostics."""

    transported = []
    successful = 0

    for detection in detections:
        new_detection = detection.copy()
        new_box, diagnostics = transport_box_with_optical_flow(
            source_gray,
            current_gray,
            detection["box"],
        )
        new_detection["box"] = new_box
        new_detection["motion_compensated"] = diagnostics["success"]
        transported.append(new_detection)
        successful += int(diagnostics["success"])

    return transported, {
        "successful": successful,
        "total": len(detections),
    }
