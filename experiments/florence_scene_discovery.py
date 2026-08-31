"""Evaluate Florence-2 as an automatic scene-vocabulary proposer.

This experiment is deliberately isolated from BHASKARA's production detector.
It samples selected video keyframes, asks Florence-2 for object detections and
dense region captions, and writes the raw evidence plus a deduplicated list of
candidate labels. Grounding DINO can later localize those dynamic labels only
after this experiment proves useful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


TASKS = ("<OD>", "<DENSE_REGION_CAPTION>")


def scene_difference(previous: np.ndarray, current: np.ndarray) -> float:
    """Return a lighting-resistant HSV histogram distance in the range 0..1."""
    def histogram(frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        value = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        return cv2.normalize(value, value).flatten()

    return float(cv2.compareHist(
        histogram(previous), histogram(current), cv2.HISTCMP_BHATTACHARYYA
    ))


def select_scene_keyframes(
    video_path: str,
    sample_interval: int = 15,
    change_threshold: float = 0.32,
    maximum_gap_seconds: float = 5.0,
    maximum_frame: int | None = None,
) -> list[int]:
    """Select changed views plus periodic coverage frames from a video."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if maximum_frame is not None:
        frame_total = min(frame_total, max(0, maximum_frame + 1))
    interval = max(1, int(sample_interval))
    maximum_gap = max(interval, int(round(fps * maximum_gap_seconds)))
    selected: list[int] = []
    previous = None
    last_selected = -maximum_gap

    for frame_index in range(0, frame_total, interval):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        changed = previous is not None and scene_difference(previous, frame) >= change_threshold
        if not selected or changed or frame_index - last_selected >= maximum_gap:
            selected.append(frame_index)
            last_selected = frame_index
        previous = frame
    capture.release()

    final_frame = frame_total - 1
    if final_frame >= 0 and (not selected or final_frame - selected[-1] >= interval):
        selected.append(final_frame)
    return selected


def normalize_candidate_label(value: str) -> str | None:
    """Turn a Florence label/caption into a stable prompt candidate."""
    cleaned = " ".join(str(value).lower().strip().split())
    cleaned = cleaned.strip(" .,:;|[]()")
    if not cleaned or len(cleaned) > 80:
        return None
    return cleaned


def labels_from_florence_result(result: Any) -> list[str]:
    """Collect labels/captions from Florence's nested post-processed output."""
    collected: list[str] = []

    def visit(value: Any, field: str | None = None) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, str(key).lower())
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested, field)
        elif isinstance(value, str) and field in {"labels", "label", "captions", "caption"}:
            normalized = normalize_candidate_label(value)
            if normalized and normalized not in collected:
                collected.append(normalized)

    visit(result)
    return collected


def read_video_frame(video_path: str, frame_index: int) -> Image.Image:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Unable to read frame {frame_index} from {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_florence(model_id: str, device: str):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    options = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": "eager",
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **options)
    except ValueError as error:
        # Florence's official Transformers 4.x remote implementation registers
        # itself as a causal language model; newer native runtimes register the
        # same architecture as image-text-to-text.
        if "Unrecognized configuration class" not in str(error):
            raise
        model = AutoModelForCausalLM.from_pretrained(model_id, **options)
    model = model.to(device)
    model.eval()
    return processor, model, dtype


def run_task(processor, model, image: Image.Image, task: str, device: str, dtype):
    inputs = processor(text=task, images=image, return_tensors="pt")
    prepared = {}
    for key, value in inputs.items():
        value = value.to(device)
        if key == "pixel_values":
            value = value.to(dtype)
        prepared[key] = value
    with torch.inference_mode():
        generated = model.generate(
            **prepared,
            max_new_tokens=768,
            num_beams=3,
            do_sample=False,
        )
    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text,
        task=task,
        image_size=image.size,
    )
    return {"generated_text": text, "parsed": parsed}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Florence-2 scene discovery experiment")
    parser.add_argument("--video", default="videos/reid_test.mp4")
    parser.add_argument("--frames", type=int, nargs="+", default=[400, 600, 847])
    parser.add_argument(
        "--auto-scenes",
        action="store_true",
        help="select changed views and periodic coverage frames automatically",
    )
    parser.add_argument("--scene-sample-interval", type=int, default=15)
    parser.add_argument("--scene-change-threshold", type=float, default=0.32)
    parser.add_argument("--scene-maximum-gap-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-frame", type=int, default=None)
    parser.add_argument("--model", default="microsoft/Florence-2-base")
    parser.add_argument("--output", default="outputs/florence_scene_discovery/report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model, dtype = load_florence(args.model, device)
    frame_reports = []
    all_candidates: list[str] = []

    frame_indices = args.frames
    if args.auto_scenes:
        frame_indices = select_scene_keyframes(
            args.video,
            sample_interval=args.scene_sample_interval,
            change_threshold=args.scene_change_threshold,
            maximum_gap_seconds=args.scene_maximum_gap_seconds,
            maximum_frame=args.maximum_frame,
        )
        print("Automatically selected Florence frames:", frame_indices)

    for frame_index in frame_indices:
        image = read_video_frame(args.video, frame_index)
        task_results = {}
        frame_candidates: list[str] = []
        for task in TASKS:
            result = run_task(processor, model, image, task, device, dtype)
            task_results[task] = result
            for label in labels_from_florence_result(result["parsed"]):
                if label not in frame_candidates:
                    frame_candidates.append(label)
                if label not in all_candidates:
                    all_candidates.append(label)
        frame_reports.append({
            "frame_index": frame_index,
            "candidates": frame_candidates,
            "tasks": task_results,
        })

    report = {
        "experiment": "florence-scene-discovery",
        "video": args.video,
        "model": args.model,
        "device": device,
        "frames": frame_reports,
        "candidate_vocabulary": all_candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Florence scene report: {output}")
    print("Candidate vocabulary:", all_candidates)


if __name__ == "__main__":
    main()
