# Florence-2 scene discovery

This is an isolated experiment. It does not alter BHASKARA's production
Grounding DINO, SigLIP, SAM 2, tracking, or memory paths.

Florence-2 currently requires a Transformers 4.x compatibility runtime while
BHASKARA's main environment uses Transformers 5.x. Keep the runtimes separate:

```powershell
python -m venv .venv-florence
```

Add the main environment's `site-packages` directory to the isolated runtime
with a local `.pth` file so it can reuse Torch/CUDA, then install:

```powershell
.\.venv-florence\Scripts\python.exe -m pip install --no-deps transformers==4.49.0 tokenizers==0.21.4 huggingface-hub==0.36.0 timm==1.0.22 einops==0.8.1
```

Run selected keyframes:

```powershell
.\.venv-florence\Scripts\python.exe -m experiments.florence_scene_discovery --video videos\reid_test.mp4 --frames 400 600 847
```

Or let BHASKARA select changed views plus periodic coverage frames and run the
complete Florence -> SigLIP -> SAM 2 flow with one command from the main venv:

```powershell
python -m experiments.sam2_video_prototype --video videos\room.mp4 --output outputs\sam2_auto_scene --auto-scene-discovery
```

The Florence model runs in `.venv-florence` as a subprocess and exits before
Grounding DINO/SigLIP/SAM 2 load, preventing incompatible Transformers versions
and the models' peak GPU memory from overlapping. Use
`--scene-sample-interval`, `--scene-change-threshold`, and
`--scene-maximum-gap-seconds` only when tuning keyframe coverage.

The experiment records Florence's object-detection and dense-region-caption
outputs plus a deduplicated candidate vocabulary. Candidates are evidence for
a future dynamic Grounding DINO prompt; they are not trusted memory labels.
