"""Render a SAM 2 prototype JSON report as an annotated MP4."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


COLORS = ((66, 214, 245), (80, 220, 100), (240, 120, 70), (190, 90, 230))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="videos/room.mp4")
    parser.add_argument("--report", default="outputs/sam2_quick/report.json")
    parser.add_argument("--output", default="outputs/sam2_quick/preview.mp4")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    labels = {item["track_id"]: item["initial_label"] for item in report["tracks"]}
    statuses = {item["track_id"]: item.get("status", "confirmed") for item in report["tracks"]}
    memory_ids = {item["track_id"]: item.get("memory_id") for item in report["tracks"]}
    observations = defaultdict(list)
    for observation in report["observations"]:
        observations[observation["frame_index"]].append(observation)
    possible_observations = defaultdict(list)
    for observation in report.get("possible_observations", []):
        possible_observations[observation["frame_index"]].append(observation)

    capture = cv2.VideoCapture(args.video)
    width, height = report["frame_size"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(report["fps"]),
        (int(width), int(height)),
    )

    frame_index = 0
    while frame_index < report["frame_count"]:
        ok, frame = capture.read()
        if not ok:
            break
        visible_track_ids = []
        for possible in possible_observations[frame_index]:
            x1, y1, x2, y2 = possible["box"]
            color = (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"POSSIBLE {possible['object']} - partial view",
                (x1 + 5, max(28, y1 + 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        for observation in observations[frame_index]:
            if statuses[observation["track_id"]] == "discarded":
                continue
            if not observation["visible"]:
                continue
            track_id = observation["track_id"]
            x1, y1, x2, y2 = observation["box"]
            color = COLORS[(track_id - 1) % len(COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame,
                f"M{memory_ids[track_id]} / T{track_id}" if memory_ids[track_id] else f"T{track_id}",
                (x1 + 5, min(y2 - 5, y1 + 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3,
            )
            visible_track_ids.append(track_id)
        if visible_track_ids:
            panel_height = 12 + 30 * len(visible_track_ids)
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, 8), (300, panel_height), (12, 12, 12), -1)
            cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
            for row, track_id in enumerate(sorted(visible_track_ids)):
                color = COLORS[(track_id - 1) % len(COLORS)]
                cv2.putText(
                    frame,
                    (
                        f"{'TENTATIVE ' if statuses[track_id] == 'tentative' else ''}"
                        f"M{memory_ids[track_id]} / T{track_id}: {labels[track_id]}"
                        if memory_ids[track_id]
                        else f"TENTATIVE T{track_id}: {labels[track_id]}"
                    ),
                    (18, 34 + row * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()
    print(f"Preview written: {output_path}")


if __name__ == "__main__":
    main()
