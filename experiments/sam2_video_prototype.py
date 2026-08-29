"""Offline Grounding DINO -> SAM 2 mask-tracking experiment.

This deliberately does not modify BHASKARA's production tracker or memory.
It discovers objects on one keyframe, assigns immutable physical track IDs,
and asks SAM 2 to propagate their masks through the benchmark video.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from detection.grounding_detector import detect_objects
from tracking.sam2_tracking import (
    consensus_label,
    duplicates_existing_track,
    eligible_new_detection,
    label_candidates,
    load_predictor,
    mask_observation,
    match_detections_to_tracks,
    tentative_track_action,
    semantic_promotion_candidates,
    semantic_promotion_approved,
)


def resolve_detection_labels(frame: np.ndarray, detections: list[dict]) -> list[dict]:
    """Canonicalize detector phrases; use SigLIP only for multi-label phrases."""
    resolved = []
    for detection in detections:
        candidates = label_candidates(detection["object"])
        if not candidates:
            continue
        selected = candidates[0]
        if len(candidates) > 1:
            from verification.verifier import verify_candidates

            x1, y1, x2, y2 = detection["box"]
            crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                continue
            result = verify_candidates(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                candidates,
            )
            if result is None or result.get("margin", 1.0) < 0.10:
                continue
            selected = result["best_label"]
        resolved.append({**detection, "raw_object": detection["object"], "object": selected})
    return resolved


def verify_semantic_promotion(frame: np.ndarray, detection: dict) -> dict:
    """Verify high-risk labels before a tentative physical track is promoted."""
    candidates = semantic_promotion_candidates(detection["object"])
    if not candidates:
        return {"approved": True, "mode": "low_risk"}

    from verification.verifier import verify_candidates

    x1, y1, x2, y2 = detection["box"]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return {"approved": False, "mode": "invalid_crop"}
    result = verify_candidates(
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
        list(candidates),
    )
    if result is None:
        return {"approved": False, "mode": "no_result"}
    approved = semantic_promotion_approved(detection["object"], result)
    return {
        "approved": approved,
        "mode": "siglip",
        "best_label": result["best_label"],
        "best_score": result["best_score"],
        "margin": result.get("margin", 0.0),
        "scores": result["scores"],
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BHASKARA SAM 2 tracking prototype")
    parser.add_argument("--video", default="videos/room.mp4")
    parser.add_argument("--output", default="outputs/sam2_prototype")
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--minimum-mask-area", type=int, default=16)
    parser.add_argument("--max-objects", type=int, default=20)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="limit extracted frames for a quick validation run",
    )
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=30,
        help="Grounding DINO refresh interval; zero disables refresh",
    )
    parser.add_argument("--refresh-min-iou", type=float, default=0.30)
    parser.add_argument("--new-track-confidence", type=float, default=0.30)
    return parser.parse_args()


def extract_frames(
    video_path: str,
    destination: Path,
    max_frames: int | None = None,
) -> tuple[int, float, tuple[int, int]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = 0
    size = (0, 0)
    while True:
        if max_frames is not None and frame_count >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        size = (width, height)
        if not cv2.imwrite(str(destination / f"{frame_count:06d}.jpg"), frame):
            raise RuntimeError(f"Unable to write extracted frame {frame_count}")
        frame_count += 1
    capture.release()
    if frame_count == 0:
        raise RuntimeError("Video contains no readable frames")
    return frame_count, fps, size


def read_prompt_frame(frame_directory: Path, frame_index: int) -> np.ndarray:
    frame = cv2.imread(str(frame_directory / f"{frame_index:06d}.jpg"))
    if frame is None:
        raise ValueError(f"Prompt frame {frame_index} is outside the video")
    return frame


def main() -> None:
    args = parse_arguments()
    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="bhaskara_sam2_") as temporary:
        frame_directory = Path(temporary)
        frame_count, fps, frame_size = extract_frames(
            args.video,
            frame_directory,
            max_frames=args.max_frames,
        )
        prompt_frame = read_prompt_frame(frame_directory, args.prompt_frame)
        detections = resolve_detection_labels(
            prompt_frame,
            detect_objects(prompt_frame),
        )[: args.max_objects]
        if not detections:
            raise RuntimeError("Grounding DINO found no objects on the prompt frame")

        predictor = load_predictor(args.model, device)
        state = predictor.init_state(video_path=str(frame_directory))
        labels: dict[int, str] = {}
        confidences: dict[int, float] = {}
        label_votes: dict[int, Counter[str]] = {}
        confirmations: dict[int, int] = {}
        track_status: dict[int, str] = {}
        missed_refreshes: dict[int, int] = {}
        for track_id, detection in enumerate(detections, start=1):
            labels[track_id] = detection["object"]
            confidences[track_id] = float(detection["confidence"])
            label_votes[track_id] = Counter({detection["object"]: 1})
            confirmations[track_id] = 1
            track_status[track_id] = "confirmed"
            missed_refreshes[track_id] = 0
            box = np.asarray(detection["box"], dtype=np.float32)
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=args.prompt_frame,
                obj_id=track_id,
                box=box,
            )

        next_track_id = len(labels) + 1
        observations = []
        visible_frames = {track_id: 0 for track_id in labels}
        refresh_events = []
        semantic_events = []
        final_tentatives_discarded = 0
        with torch.inference_mode():
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
            with autocast:
                interval = args.refresh_interval if args.refresh_interval > 0 else frame_count
                chunk_start = args.prompt_frame
                while chunk_start < frame_count:
                    chunk_end = min(frame_count - 1, chunk_start + interval - 1)
                    end_boxes = {}
                    for frame_index, object_ids, mask_logits in predictor.propagate_in_video(
                        state,
                        start_frame_idx=chunk_start,
                        max_frame_num_to_track=chunk_end - chunk_start,
                    ):
                        for object_id, logits in zip(object_ids, mask_logits):
                            track_id = int(object_id)
                            observation = mask_observation(
                                track_id,
                                int(frame_index),
                                (logits > 0.0).detach().cpu().numpy(),
                                minimum_area=args.minimum_mask_area,
                            )
                            visible_frames.setdefault(track_id, 0)
                            visible_frames[track_id] += int(observation.visible)
                            if observation.visible and int(frame_index) == chunk_end:
                                end_boxes[track_id] = observation.box
                            observations.append(
                                {
                                    "track_id": track_id,
                                    "frame_index": observation.frame_index,
                                    "box": observation.box,
                                    "mask_area": observation.area,
                                    "visible": observation.visible,
                                }
                            )

                    if chunk_end >= frame_count - 1 or args.refresh_interval <= 0:
                        break

                    refresh_frame = read_prompt_frame(frame_directory, chunk_end)
                    refreshed = resolve_detection_labels(refresh_frame, detect_objects(refresh_frame))
                    assignments, unmatched = match_detections_to_tracks(
                        end_boxes,
                        refreshed,
                        minimum_iou=args.refresh_min_iou,
                    )
                    event = {
                        "frame_index": chunk_end,
                        "detections": len(refreshed),
                        "matched": len(assignments),
                        "enrolled": 0,
                        "duplicate_enrollment_blocked": 0,
                        "promoted": 0,
                        "discarded": 0,
                        "semantic_rejected": 0,
                    }
                    matched_track_ids = set()
                    semantic_removals = []
                    for track_id, detection_index in assignments.items():
                        detection = refreshed[detection_index]
                        newest = detection["object"]
                        if (
                            track_status[track_id] == "tentative"
                            and newest != labels[track_id]
                        ):
                            unmatched.append(detection_index)
                            continue
                        matched_track_ids.add(track_id)
                        label_votes[track_id][newest] += 1
                        confirmations[track_id] += 1
                        missed_refreshes[track_id] = 0
                        labels[track_id] = consensus_label(label_votes[track_id], newest)
                        confidences[track_id] = max(
                            confidences[track_id], float(detection["confidence"])
                        )
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=chunk_end,
                            obj_id=track_id,
                            box=np.asarray(detection["box"], dtype=np.float32),
                        )
                        if track_status[track_id] == "tentative":
                            action = tentative_track_action(
                                confirmations[track_id], True, 0
                            )
                            if action == "promote":
                                semantic = verify_semantic_promotion(
                                    refresh_frame, detection
                                )
                                semantic_events.append({
                                    "frame_index": chunk_end,
                                    "track_id": track_id,
                                    "detector_label": newest,
                                    **semantic,
                                })
                                if semantic["approved"]:
                                    track_status[track_id] = "confirmed"
                                    event["promoted"] += 1
                                else:
                                    track_status[track_id] = "discarded"
                                    semantic_removals.append(track_id)
                                    event["semantic_rejected"] += 1

                    for track_id in semantic_removals:
                        predictor.remove_object(
                            state, track_id, strict=True, need_output=False
                        )
                        end_boxes.pop(track_id, None)

                    for track_id in list(end_boxes):
                        if (
                            track_status.get(track_id) != "tentative"
                            or track_id in matched_track_ids
                        ):
                            continue
                        missed_refreshes[track_id] += 1
                        action = tentative_track_action(
                            confirmations[track_id],
                            False,
                            missed_refreshes[track_id],
                        )
                        if action == "discard":
                            predictor.remove_object(
                                state, track_id, strict=True, need_output=False
                            )
                            track_status[track_id] = "discarded"
                            end_boxes.pop(track_id, None)
                            event["discarded"] += 1

                    for detection_index in unmatched:
                        detection = refreshed[detection_index]
                        if duplicates_existing_track(detection, end_boxes, labels):
                            event["duplicate_enrollment_blocked"] += 1
                            continue
                        if not eligible_new_detection(
                            detection, frame_size, args.new_track_confidence
                        ):
                            continue
                        track_id = next_track_id
                        next_track_id += 1
                        labels[track_id] = detection["object"]
                        confidences[track_id] = float(detection["confidence"])
                        label_votes[track_id] = Counter({detection["object"]: 1})
                        confirmations[track_id] = 1
                        track_status[track_id] = "tentative"
                        missed_refreshes[track_id] = 0
                        visible_frames[track_id] = 0
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=chunk_end,
                            obj_id=track_id,
                            box=np.asarray(detection["box"], dtype=np.float32),
                        )
                        event["enrolled"] += 1
                    refresh_events.append(event)
                    chunk_start = chunk_end + 1

                for track_id, status in list(track_status.items()):
                    if status == "tentative":
                        track_status[track_id] = "discarded"
                        final_tentatives_discarded += 1

    report = {
        "prototype": "grounding-dino-sam2-offline",
        "video": args.video,
        "model": args.model,
        "device": device,
        "frame_count": frame_count,
        "fps": fps,
        "frame_size": frame_size,
        "prompt_frame": args.prompt_frame,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "tracks": [
            {
                "track_id": track_id,
                "initial_label": labels[track_id],
                "detector_confidence": confidences[track_id],
                "visible_frames": visible_frames[track_id],
                "confirmations": confirmations[track_id],
                "label_votes": dict(label_votes[track_id]),
                "status": track_status[track_id],
            }
            for track_id in sorted(labels)
        ],
        "observations": observations,
        "refresh_events": refresh_events,
        "semantic_events": semantic_events,
        "final_tentatives_discarded": final_tentatives_discarded,
    }
    report_path = output_directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SAM 2 prototype report: {report_path}")
    print(f"Tracks: {len(labels)} | Frames: {frame_count} | Device: {device}")


if __name__ == "__main__":
    main()
