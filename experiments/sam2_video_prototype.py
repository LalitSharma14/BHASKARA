"""Offline Grounding DINO -> SAM 2 mask-tracking experiment.

This deliberately does not modify BHASKARA's production tracker or memory.
It discovers objects on one keyframe, assigns immutable physical track IDs,
and asks SAM 2 to propagate their masks through the benchmark video.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from detection.grounding_detector import detect_objects
from detection.grounding_detector import SEARCH_CLASSES
from detection.scene_vocabulary import (
    dynamic_verification_candidates,
    load_scene_proposals,
    load_scene_vocabulary,
    merge_verified_scene_proposals,
    refined_scene_label,
    possible_scene_label,
    scene_label_requires_parent_context,
)
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
    confirmed_track_action,
    requires_edge_safe_enrollment,
    touches_frame_edge,
    find_persistent_identity,
    update_persistent_identity,
)
from tracking.appearance_quality import describe_crop


def resolve_detection_labels(
    frame: np.ndarray,
    detections: list[dict],
    extra_labels=(),
) -> list[dict]:
    """Canonicalize detector phrases; use SigLIP only for multi-label phrases."""
    resolved = []
    for detection in detections:
        candidates = label_candidates(detection["object"], extra_labels=extra_labels)
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


def verify_semantic_promotion(
    frame: np.ndarray,
    detection: dict,
    dynamic_labels=(),
) -> dict:
    """Verify high-risk labels before a tentative physical track is promoted."""
    candidates = semantic_promotion_candidates(detection["object"])
    if not candidates and detection["object"] in dynamic_labels:
        candidates = dynamic_verification_candidates(detection["object"])
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


def verify_scene_proposals(frame: np.ndarray, proposals: list[dict]):
    """Semantically verify Florence regions before they can replace detections."""
    if not proposals:
        return [], []
    from verification.verifier import verify_candidates

    verified = []
    events = []
    height, width = frame.shape[:2]
    for proposal in proposals:
        x1, y1, x2, y2 = proposal["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        candidates = dynamic_verification_candidates(proposal["object"])
        result = verify_candidates(
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
            candidates,
        )
        # Florence provides independent localization evidence, so the SigLIP
        # gate may use a slightly lower score/margin than detector-only track
        # promotion while still requiring the exact proposed label to win.
        final_label = refined_scene_label(proposal["object"], result)
        semantically_approved = (
            final_label is not None
            and float(result.get("best_score", 0.0)) >= 0.50
            and float(result.get("margin", 0.0)) >= 0.10
        )
        component_only = (
            semantically_approved
            and scene_label_requires_parent_context(final_label)
        )
        approved = semantically_approved and not component_only
        possible_label = None if approved else possible_scene_label(
            proposal["object"], proposal.get("raw_object", proposal["object"])
        )
        events.append({
            "object": proposal["object"],
            "raw_object": proposal.get("raw_object", proposal["object"]),
            "box": proposal["box"],
            "approved": approved,
            "status": "confirmed" if approved else (
                "possible" if possible_label else "rejected"
            ),
            "possible_label": possible_label,
            "reason": "component_requires_parent_context" if component_only else None,
            "result": result,
        })
        if approved:
            verified.append({
                **proposal,
                "object": final_label,
                "discovered_object": proposal["object"],
                "box": (x1, y1, x2, y2),
                "confidence": float(result["best_score"]),
            })
    return verified, events


def add_appearance_embeddings(
    frame: np.ndarray,
    detections: list[dict],
    frame_size: tuple[int, int],
) -> list[dict]:
    """Attach one batched set of tight-crop embeddings and quality metadata."""
    prepared = [dict(detection) for detection in detections]
    crops = []
    crop_indices = []
    width, height = frame_size
    for index, detection in enumerate(prepared):
        quality = describe_crop(detection["box"], width, height)
        detection.update({
            "appearance_aspect_ratio": quality["aspect_ratio"],
            "appearance_tiny": quality["tiny"],
            "appearance_touches_edge": quality["touches_edge"],
            "appearance_quality": quality["quality"],
            "appearance_embedding": None,
        })
        # Tiny and boundary-truncated crops may help detection, but their
        # resized pixels are not reliable evidence of physical identity.
        if not quality["valid"] or quality["tiny"] or quality["touches_edge"]:
            continue
        x1, y1, x2, y2 = quality["box"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        crop_indices.append(index)

    if crops:
        from verification.verifier import get_image_embeddings

        for index, embedding in zip(crop_indices, get_image_embeddings(crops)):
            prepared[index]["appearance_embedding"] = embedding
    return prepared


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
        "--scene-vocabulary-report",
        action="append",
        default=[],
        help="Florence scene report; may be supplied more than once",
    )
    parser.add_argument(
        "--auto-scene-discovery",
        action="store_true",
        help="automatically run isolated Florence discovery before tracking",
    )
    parser.add_argument(
        "--florence-python",
        default=str(Path(".venv-florence") / "Scripts" / "python.exe"),
        help="Python executable for the isolated Florence Transformers 4.x runtime",
    )
    parser.add_argument("--scene-sample-interval", type=int, default=15)
    parser.add_argument("--scene-change-threshold", type=float, default=0.32)
    parser.add_argument("--scene-maximum-gap-seconds", type=float, default=5.0)
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=30,
        help="Grounding DINO refresh interval; zero disables refresh",
    )
    parser.add_argument("--refresh-min-iou", type=float, default=0.30)
    parser.add_argument("--new-track-confidence", type=float, default=0.30)
    parser.add_argument(
        "--confirmed-missed-refreshes",
        type=int,
        default=2,
        help="retire confirmed SAM state after this many missed detector refreshes",
    )
    parser.add_argument(
        "--keep-sam-state-on-gpu",
        action="store_true",
        help="faster for short clips but may exhaust VRAM on complete videos",
    )
    return parser.parse_args()


def run_automatic_scene_discovery(args, output_directory: Path) -> str:
    """Run Florence out-of-process so its Transformers version stays isolated."""
    florence_python = Path(args.florence_python).resolve()
    if not florence_python.is_file():
        raise RuntimeError(
            f"Florence runtime not found: {florence_python}. "
            "Create .venv-florence as documented in experiments/FLORENCE_SCENE_DISCOVERY.md."
        )
    report_path = (output_directory / "automatic_scene_report.json").resolve()
    command = [
        str(florence_python),
        "-m", "experiments.florence_scene_discovery",
        "--video", str(Path(args.video).resolve()),
        "--output", str(report_path),
        "--auto-scenes",
        "--scene-sample-interval", str(args.scene_sample_interval),
        "--scene-change-threshold", str(args.scene_change_threshold),
        "--scene-maximum-gap-seconds", str(args.scene_maximum_gap_seconds),
    ]
    if args.max_frames is not None:
        command.extend(["--maximum-frame", str(max(0, args.max_frames - 1))])
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
    return str(report_path)


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
    scene_report_paths = list(args.scene_vocabulary_report)
    if args.auto_scene_discovery:
        scene_report_paths.append(run_automatic_scene_discovery(args, output_directory))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="bhaskara_sam2_") as temporary:
        frame_directory = Path(temporary)
        frame_count, fps, frame_size = extract_frames(
            args.video,
            frame_directory,
            max_frames=args.max_frames,
        )
        dynamic_labels = load_scene_vocabulary(scene_report_paths)
        scene_proposals = load_scene_proposals(scene_report_paths)
        proposal_frames = sorted(
            frame_index
            for frame_index in scene_proposals
            if frame_index > args.prompt_frame
        )
        detector_search_classes = SEARCH_CLASSES
        prompt_frame = read_prompt_frame(frame_directory, args.prompt_frame)
        detections = resolve_detection_labels(
            prompt_frame,
            detect_objects(prompt_frame, search_classes=detector_search_classes),
            extra_labels=dynamic_labels,
        )
        prompt_proposals, prompt_scene_events = verify_scene_proposals(
            prompt_frame, scene_proposals.get(args.prompt_frame, [])
        )
        detections = merge_verified_scene_proposals(
            detections, prompt_proposals
        )[: args.max_objects]
        detections = add_appearance_embeddings(prompt_frame, detections, frame_size)
        if not detections:
            raise RuntimeError("Grounding DINO found no objects on the prompt frame")

        predictor = load_predictor(args.model, device)
        offload_sam_state = not args.keep_sam_state_on_gpu
        state = predictor.init_state(
            video_path=str(frame_directory),
            offload_video_to_cpu=offload_sam_state,
            offload_state_to_cpu=offload_sam_state,
            async_loading_frames=offload_sam_state,
        )
        labels: dict[int, str] = {}
        confidences: dict[int, float] = {}
        label_votes: dict[int, Counter[str]] = {}
        confirmations: dict[int, int] = {}
        track_status: dict[int, str] = {}
        missed_refreshes: dict[int, int] = {}
        authoritative_labels: dict[int, str] = {}
        track_appearances: dict[int, dict] = {}
        track_memory_ids: dict[int, int] = {}
        physical_identities: dict[int, dict] = {}
        identity_events = []
        scene_proposal_events = [
            {"frame_index": args.prompt_frame, **event}
            for event in prompt_scene_events
        ]
        next_memory_id = 1

        def assign_persistent_identity(
            track_id: int,
            detection: dict,
            frame_index: int,
            allow_reuse: bool,
        ) -> int:
            nonlocal next_memory_id
            observation = {**detection, "object": labels[track_id]}
            refined_from_generic = (
                detection.get("source") == "florence"
                and detection.get("discovered_object")
                and detection.get("discovered_object") != observation["object"]
            )
            compatible_identity_labels = ()
            if refined_from_generic:
                compatible_identity_labels = dynamic_verification_candidates(
                    detection["discovered_object"]
                )
            if allow_reuse:
                match, diagnostics = find_persistent_identity(
                    observation,
                    physical_identities,
                    return_diagnostics=True,
                    compatible_labels=compatible_identity_labels,
                )
            else:
                match = None
                diagnostics = {"reason": "initial_identity"}
            if match is None:
                memory_id = next_memory_id
                next_memory_id += 1
                event = "created"
            else:
                memory_id = match["memory_id"]
                event = "reidentified"
            track_memory_ids[track_id] = memory_id
            if match is not None and refined_from_generic:
                # The independently localized and verified specific class is
                # authoritative for every fragment already sharing this
                # physical identity. This repairs memory instead of creating
                # a parallel AC and medicine-box identity.
                for previous_track_id, previous_memory_id in track_memory_ids.items():
                    if previous_track_id == track_id or previous_memory_id != memory_id:
                        continue
                    corrected_label = observation["object"]
                    label_votes[previous_track_id][corrected_label] = max(
                        label_votes[previous_track_id][corrected_label],
                        max(label_votes[previous_track_id].values()) + 1,
                    )
                    labels[previous_track_id] = corrected_label
                    authoritative_labels[previous_track_id] = corrected_label
            physical_identities[memory_id] = update_persistent_identity(
                physical_identities.get(memory_id), observation
            )
            identity_events.append({
                "frame_index": frame_index,
                "track_id": track_id,
                "memory_id": memory_id,
                "event": event,
                "similarity": match.get("similarity") if match else None,
                "second_best_similarity": (
                    match.get("second_best_similarity") if match else None
                ),
                "margin": match.get("margin") if match else None,
                "match_diagnostics": diagnostics,
            })
            return memory_id

        for track_id, detection in enumerate(detections, start=1):
            labels[track_id] = detection["object"]
            confidences[track_id] = float(detection["confidence"])
            label_votes[track_id] = Counter({detection["object"]: 1})
            confirmations[track_id] = 1
            track_status[track_id] = "confirmed"
            missed_refreshes[track_id] = 0
            if detection.get("source") == "florence":
                authoritative_labels[track_id] = detection["object"]
            track_appearances[track_id] = detection
            assign_persistent_identity(
                track_id, detection, args.prompt_frame, allow_reuse=False
            )
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
                    upcoming_proposals = [
                        frame_index
                        for frame_index in proposal_frames
                        if chunk_start <= frame_index < chunk_end
                    ]
                    if upcoming_proposals:
                        chunk_end = upcoming_proposals[0]
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
                    refreshed = resolve_detection_labels(
                        refresh_frame,
                        detect_objects(
                            refresh_frame, search_classes=detector_search_classes
                        ),
                        extra_labels=dynamic_labels,
                    )
                    verified_proposals, proposal_events = verify_scene_proposals(
                        refresh_frame, scene_proposals.get(chunk_end, [])
                    )
                    scene_proposal_events.extend(
                        {"frame_index": chunk_end, **proposal_event}
                        for proposal_event in proposal_events
                    )
                    refreshed = merge_verified_scene_proposals(
                        refreshed, verified_proposals
                    )
                    refreshed = add_appearance_embeddings(
                        refresh_frame, refreshed, frame_size
                    )
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
                        "retired": 0,
                    }
                    matched_track_ids = set()
                    semantic_removals = []
                    for track_id, detection_index in assignments.items():
                        detection = refreshed[detection_index]
                        newest = detection["object"]
                        authoritative = authoritative_labels.get(track_id)
                        if (
                            authoritative is not None
                            and detection.get("source") != "florence"
                            and newest != authoritative
                        ):
                            # Use the detector's fresh geometry while retaining
                            # the independently verified landmark semantics.
                            newest = authoritative
                            detection = {**detection, "object": authoritative}
                        if (
                            track_status[track_id] == "tentative"
                            and newest != labels[track_id]
                        ):
                            unmatched.append(detection_index)
                            continue
                        matched_track_ids.add(track_id)
                        if (
                            detection.get("source") == "florence"
                            and newest != labels[track_id]
                        ):
                            # A Florence region whose label independently won
                            # the SigLIP confusion check is stronger than one
                            # ordinary detector vote. Preserve the physical
                            # track but let the corrected label take the lead.
                            label_votes[track_id][newest] = max(
                                label_votes[track_id][newest],
                                max(label_votes[track_id].values()),
                            )
                            authoritative_labels[track_id] = newest
                        elif detection.get("source") == "florence":
                            authoritative_labels[track_id] = newest
                        label_votes[track_id][newest] += 1
                        confirmations[track_id] += 1
                        missed_refreshes[track_id] = 0
                        labels[track_id] = consensus_label(label_votes[track_id], newest)
                        confidences[track_id] = max(
                            confidences[track_id], float(detection["confidence"])
                        )
                        track_appearances[track_id] = detection
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
                                    refresh_frame, detection, dynamic_labels
                                )
                                semantic_events.append({
                                    "frame_index": chunk_end,
                                    "track_id": track_id,
                                    "detector_label": newest,
                                    **semantic,
                                })
                                if semantic["approved"]:
                                    track_status[track_id] = "confirmed"
                                    assign_persistent_identity(
                                        track_id, detection, chunk_end, allow_reuse=True
                                    )
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

                    active_ids = list(state["obj_ids"])
                    for track_id in active_ids:
                        if track_status.get(track_id) != "confirmed":
                            continue
                        if track_id in matched_track_ids:
                            missed_refreshes[track_id] = 0
                            memory_id = track_memory_ids.get(track_id)
                            if memory_id is not None:
                                physical_identities[memory_id] = update_persistent_identity(
                                    physical_identities.get(memory_id),
                                    {**track_appearances[track_id], "object": labels[track_id]},
                                )
                            continue
                        missed_refreshes[track_id] += 1
                        action = confirmed_track_action(
                            False,
                            missed_refreshes[track_id],
                            maximum_missed_refreshes=args.confirmed_missed_refreshes,
                        )
                        if action == "retire":
                            predictor.remove_object(
                                state, track_id, strict=True, need_output=False
                            )
                            track_status[track_id] = "retired"
                            end_boxes.pop(track_id, None)
                            event["retired"] += 1

                    for detection_index in unmatched:
                        detection = refreshed[detection_index]
                        if duplicates_existing_track(detection, end_boxes, labels):
                            event["duplicate_enrollment_blocked"] += 1
                            continue
                        if not eligible_new_detection(
                            detection, frame_size, args.new_track_confidence
                        ):
                            continue
                        if (
                            requires_edge_safe_enrollment(detection["object"])
                            and touches_frame_edge(tuple(detection["box"]), frame_size)
                        ):
                            continue
                        track_id = next_track_id
                        next_track_id += 1
                        labels[track_id] = detection["object"]
                        confidences[track_id] = float(detection["confidence"])
                        label_votes[track_id] = Counter({detection["object"]: 1})
                        scene_confirmed = detection.get("source") == "florence"
                        if scene_confirmed:
                            authoritative_labels[track_id] = detection["object"]
                        confirmations[track_id] = 2 if scene_confirmed else 1
                        track_status[track_id] = (
                            "confirmed" if scene_confirmed else "tentative"
                        )
                        missed_refreshes[track_id] = 0
                        visible_frames[track_id] = 0
                        track_appearances[track_id] = detection
                        if scene_confirmed:
                            assign_persistent_identity(
                                track_id, detection, chunk_end, allow_reuse=True
                            )
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=chunk_end,
                            obj_id=track_id,
                            box=np.asarray(detection["box"], dtype=np.float32),
                        )
                        # Include this proposal immediately so a contained or
                        # overlapping same-scan proposal cannot create another ID.
                        end_boxes[track_id] = tuple(detection["box"])
                        event["enrolled"] += 1
                    refresh_events.append(event)
                    chunk_start = chunk_end + 1

                final_tentative_ids = [
                    track_id
                    for track_id, status in track_status.items()
                    if status == "tentative"
                ]
                final_frame = None
                final_detections = []
                final_assignments = {}
                if final_tentative_ids:
                    final_frame = read_prompt_frame(frame_directory, frame_count - 1)
                    final_detections = resolve_detection_labels(
                        final_frame,
                        detect_objects(
                            final_frame, search_classes=detector_search_classes
                        ),
                        extra_labels=dynamic_labels,
                    )
                    final_proposals, final_proposal_events = verify_scene_proposals(
                        final_frame, scene_proposals.get(frame_count - 1, [])
                    )
                    scene_proposal_events.extend(
                        {"frame_index": frame_count - 1, **proposal_event}
                        for proposal_event in final_proposal_events
                    )
                    final_detections = merge_verified_scene_proposals(
                        final_detections, final_proposals
                    )
                    final_detections = add_appearance_embeddings(
                        final_frame, final_detections, frame_size
                    )
                    final_boxes = {
                        observation["track_id"]: observation["box"]
                        for observation in observations
                        if observation["track_id"] in final_tentative_ids
                        and observation["visible"]
                        and observation["frame_index"] == frame_count - 1
                    }
                    final_assignments, _ = match_detections_to_tracks(
                        final_boxes,
                        final_detections,
                        minimum_iou=args.refresh_min_iou,
                    )

                for track_id, status in list(track_status.items()):
                    if status == "tentative":
                        detection_index = final_assignments.get(track_id)
                        if (
                            detection_index is not None
                            and final_detections[detection_index]["object"] == labels[track_id]
                        ):
                            semantic = verify_semantic_promotion(
                                final_frame,
                                final_detections[detection_index],
                                dynamic_labels,
                            )
                            semantic_events.append({
                                "frame_index": frame_count - 1,
                                "track_id": track_id,
                                "detector_label": labels[track_id],
                                "final_boundary": True,
                                **semantic,
                            })
                            if semantic["approved"]:
                                confirmations[track_id] += 1
                                label_votes[track_id][labels[track_id]] += 1
                                track_status[track_id] = "confirmed"
                                track_appearances[track_id] = final_detections[detection_index]
                                assign_persistent_identity(
                                    track_id,
                                    final_detections[detection_index],
                                    frame_count - 1,
                                    allow_reuse=True,
                                )
                                continue
                        track_status[track_id] = "discarded"
                        final_tentatives_discarded += 1

    report = {
        "prototype": "grounding-dino-sam2-offline",
        "video": args.video,
        "model": args.model,
        "device": device,
        "sam_state_offloaded_to_cpu": offload_sam_state,
        "sam_frames_loaded_asynchronously": offload_sam_state,
        "frame_count": frame_count,
        "fps": fps,
        "frame_size": frame_size,
        "prompt_frame": args.prompt_frame,
        "dynamic_scene_labels": dynamic_labels,
        "scene_proposal_events": scene_proposal_events,
        "possible_observations": [
            {
                "frame_index": event["frame_index"],
                "object": event["possible_label"],
                "raw_object": event["raw_object"],
                "box": event["box"],
                "status": "possible",
                "memory_id": None,
                "reason": event.get("reason") or "partial_object_evidence",
            }
            for event in scene_proposal_events
            if event.get("status") == "possible" and event.get("possible_label")
        ],
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
                "memory_id": track_memory_ids.get(track_id),
                "authoritative_label": authoritative_labels.get(track_id),
            }
            for track_id in sorted(labels)
        ],
        "observations": observations,
        "refresh_events": refresh_events,
        "semantic_events": semantic_events,
        "identity_events": identity_events,
        "physical_identities": [
            {
                "memory_id": memory_id,
                "object": identity["object"],
                "appearance_views": len(identity.get("appearance_gallery", [])),
                "track_ids": sorted(
                    track_id
                    for track_id, assigned_memory_id in track_memory_ids.items()
                    if assigned_memory_id == memory_id
                ),
            }
            for memory_id, identity in sorted(physical_identities.items())
        ],
        "final_tentatives_discarded": final_tentatives_discarded,
    }
    report_path = output_directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SAM 2 prototype report: {report_path}")
    print(f"Tracks: {len(labels)} | Frames: {frame_count} | Device: {device}")


if __name__ == "__main__":
    main()
