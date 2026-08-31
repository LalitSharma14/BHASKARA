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

With the isolated Florence runtime installed, automatic scene discovery can be
included in the same command:

```powershell
python -m experiments.sam2_video_prototype --video videos/room.mp4 --output outputs/sam2_auto_scene --auto-scene-discovery
```

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

Confirmed tracks are removed from active SAM 2 state after three consecutive
detector-refresh misses. Their metadata remains in the report as `retired`, but
SAM no longer spends GPU time propagating an object that has left the view.

Before promotion, high-risk labels receive one class-specific SigLIP check
against realistic look-alikes. For example, `metal ruler` is compared with
`lanyard`, `ribbon`, and `strap`. A tentative track is removed immediately when
its detector label does not win with sufficient score and margin.

Small portable-object candidates touching a frame edge are rejected before
enrollment; this specifically prevents bottom-edge floor-tile fragments from
becoming `id card` tracks. The ID-card semantic alternatives also explicitly
include floor tile and grout patterns.

Tentative tracks created in the last partial interval receive one detector scan
on the final frame. This allows a real late-appearing object such as a fan to be
confirmed, while single-scan end-of-video candidates remain discarded.

Video tensors and historical SAM 2 mask state are offloaded to CPU by default.
This keeps enough VRAM available for Grounding DINO and SigLIP refreshes on a
6 GB GPU. `--keep-sam-state-on-gpu` is available only for short performance
experiments where the complete state is known to fit.
Complete-video runs also use SAM 2's asynchronous frame loader so hundreds of
resized video tensors are not decoded into RAM at initialization.

Combined detector phrases such as `door cabinet shelf` are treated as
ambiguous. The existing SigLIP verifier selects among the contained canonical
labels, and low-margin results are rejected instead of silently choosing the
first word.

## Persistent physical identity

SAM object IDs are tracking-session handles, not permanent object identities.
The prototype therefore reports both:

- `track_id`: the disposable SAM 2 mask track.
- `memory_id`: BHASKARA's persistent physical-object identity.

Only confirmed tracks receive a memory ID. A returning same-label track may
reuse an older memory ID when its tight-crop SigLIP embedding clears the
appearance threshold and has a clear margin over every other candidate.
Different-looking, ambiguous, missing-embedding, severe-aspect-change, tiny,
and frame-edge observations remain separate. Each identity retains a small
gallery of diverse tight-object views; context embeddings are not generated.
