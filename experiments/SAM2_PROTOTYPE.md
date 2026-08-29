# Grounding DINO + SAM 2 tracking prototype

This experiment is isolated from `detection/video_detector.py`. It uses one
Grounding DINO keyframe to initialize immutable track IDs and propagates object
masks with SAM 2.1. Its first purpose is to measure mask continuity on the same
video used by BHASKARA's deterministic tracker.

SAM 2 is intentionally not installed as a normal BHASKARA dependency yet. Meta
recommends WSL for Windows, and the model also requires a checkpoint download.

After installing SAM 2 in an isolated compatible environment, run from the
repository root:

```powershell
python -m experiments.sam2_video_prototype --video videos/room.mp4
```

Use `--max-frames 120 --max-objects 3` for a short end-to-end validation
before running all 677 frames and every initial detection.

The machine-readable result is written to
`outputs/sam2_prototype/report.json`. The report records immutable track IDs,
initial detector labels, per-frame mask boxes and areas, and visibility counts.

Render a report for visual inspection with:

```powershell
python -m experiments.render_sam2_report --report outputs/sam2_quick/report.json
```

This first experiment does not update trusted object memory and does not claim
to update trusted object memory. At each configured refresh boundary, detector
boxes are uniquely matched to visible SAM 2 masks, label votes are updated,
and the matched box is used as a corrective prompt. High-confidence unmatched
detections may be enrolled with a new immutable ID; weak, collapsed, and
near-full-frame detections are rejected. Same-label unmatched boxes are also
rejected when they substantially overlap or are contained by an existing mask
box, preventing a partial view from creating a duplicate physical ID.

An eligible unmatched detection begins as a tentative SAM 2 track. It is
promoted only after a same-label match at the following detector refresh. A
tentative track that misses that refresh is removed from SAM 2 state, preventing
single-scan false positives from accumulating inference cost. Tentative and
discarded tracks are never eligible for trusted object memory.

Before promotion, high-risk labels receive one class-specific SigLIP check
against realistic look-alikes. For example, `metal ruler` is compared with
`lanyard`, `ribbon`, and `strap`. A tentative track is removed immediately when
its detector label does not win with sufficient score and margin.

Combined detector phrases such as `door cabinet shelf` are treated as
ambiguous. The existing SigLIP verifier selects among the contained canonical
labels, and low-margin results are rejected instead of silently choosing the
first word.
