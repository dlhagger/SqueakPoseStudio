SqueakPose Studio
=================

Desktop labeling, training, and inference toolkit for small-animal (mouse) pose estimation. Built with PyQt6 and Ultralytics YOLO to streamline the full loop: annotate frames, export YOLO-format datasets, train models, and run video inference with per-frame CSV outputs.

Overview
--------
- Interactive labeling UI: draw one bounding box per frame, place ordered keypoints, mark visibility (visible / occluded / not present), and auto-clamp to image bounds. Modes: pan/zoom, bbox, keypoint, predict.
- Dataset management: keeps originals in `images_all`, YOLO labels in `labels_all`, and annotated overlays in `annotations`. Export train/val splits and a YOLO `dataset.yaml` directly from the app.
- Training launcher: choose task (pose or detection), dataset, epochs, batch size/device, and trigger `ultralytics.YOLO.train()` without leaving the UI.
- Video inference: batch process videos with a trained model; exports rich CSV metrics per detection + keypoints (normalized coords, areas, timing) to an `inference outputs` folder.
- Programmatic helper: `dataset_builder.py` generates YOLO pose `dataset.yaml` files with sensible flip indices based on keypoint names.

Repository layout
-----------------
- `squeakpose_studio.py`: main PyQt6 application (labeling, exporting, training, inference).
- `dataset_builder.py`: helper to emit YOLO pose `dataset.yaml` files.
- `images_to_label/`: drop raw frames to annotate (created on first run).
- `images_all/`, `labels_all/`, `annotations/`: accumulated originals, YOLO labels, and rendered overlays (auto-managed on Save).
- `datasets/`: created when exporting train/val splits.
- `runs/`: default Ultralytics training outputs.
- `classes.txt`, `keypoints.txt`: class and keypoint lists (defaults provided; editable).
- `fonts/`: optional UI font (Fira Sans).
- `torch_test_apple_silicon.py`: quick check that PyTorch + MPS is working on Apple Silicon.

Requirements
------------
- Python >= 3.12
- PyQt6, torch, torchvision, ultralytics (declared in `pyproject.toml`)
- OpenCV (`opencv-python`) for video inference
- Optional: Apple Silicon / CUDA for faster training and inference (device auto-selected in app)

Setup
-----
1) Create a virtualenv and install:
```
pip install -e .
```
   If using `uv`, the PyTorch indexes in `pyproject.toml` are preconfigured.

2) Ensure `classes.txt` and `keypoints.txt` exist in the project root. On first launch the app will create defaults (`mouse` class; 6 keypoints).

Running the app
---------------
```
python squeakpose_studio.py
```
The app will create `images_to_label`, `images_all`, `labels_all`, `annotations`, and `fonts` if they are missing.

Labeling workflow
-----------------
1) Add images to `images_to_label/`.
2) Launch the app and pick modes from the floating panel or shortcuts (1: pan/zoom, 2: bbox, 3: keypoint, 4: predict).
3) Draw exactly one bounding box, then place keypoints in the order defined by `keypoints.txt`. Toggle keypoint visibility states as needed.
4) Click **Save**. This writes YOLO labels to `labels_all/<image>.txt`, copies the source to `images_all/`, and renders an overlay to `annotations/`.
5) Browse with filters (All/Labeled/Unlabeled) and use the “Complete → Next Unlabeled” workflow to advance.

Dataset export & training
-------------------------
- **Export Dataset**: Splits `images_all`/`labels_all` into train/val (configurable ratio) and writes a YOLO `dataset.yaml` under `datasets/pose` (or `datasets/detect` if exporting bbox-only).
- **Train Model**: Pick a base model/config, dataset path, device, epochs, and batch size; launches Ultralytics training. Outputs land in `runs/` (Ultralytics default).

Video inference
---------------
- Load a trained model via **Load Model**.
- **Inference** prompts for a video, batch size, and runs YOLO on batches. Results are saved as CSV to an `inference outputs` folder (one level above the repo) with per-frame detection + keypoint metrics and normalized coordinates.

Programmatic helper
-------------------
Generate a YOLO pose `dataset.yaml` from Python:
```python
from dataset_builder import create_dataset_yaml

create_dataset_yaml(
    base_dir="datasets/pose",          # folder containing images/train, images/val, labels/train, labels/val
    class_names=["mouse"],
    kp_names=["nose", "head", "left_ear", "right_ear", "back", "tail_base"],
)
```
Flip indices are inferred automatically when keypoint names contain “left”/“right”.

Troubleshooting
---------------
- Device selection is automatic (CUDA → MPS → CPU). Run `python torch_test_apple_silicon.py` to verify Apple Silicon/MPS.
- If OpenCV is missing, install `opencv-python` to enable video inference.
- For missing class/keypoint files, create or edit `classes.txt` and `keypoints.txt` at the project root.

License & notice
----------------
This code is a U.S. Government work (NIH) and in the public domain, subject to third-party license obligations (PyQt6 GPLv3/commercial, Ultralytics AGPL-3.0, PyTorch BSD-style). The repository is private and pending NIH OTT review; no commercial rights are granted at this time. See `LICENSE` and `NOTICE.txt` for details.
