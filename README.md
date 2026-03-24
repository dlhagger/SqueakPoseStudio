SqueakPose Studio
=================

Desktop labeling, training, and inference toolkit for small-animal (mouse) annotation. Built with PyQt6 and Ultralytics YOLO to streamline the full loop: annotate frames (pose and segmentation), export YOLO-format datasets, train models, and run video inference with per-frame CSV outputs.

Overview
--------
- Project-based workflow: launch into **Open Project / Create Project**, keep all assets per-project, and restore project metadata (including last active workflow).
- Dual labeling workflows from one UI:
  - **Pose Workflow (BBox + Keypoints)**: draw one box, place ordered keypoints, set visibility states, and save YOLO pose labels.
  - **Segmentation Workflow (SAM)**: place positive/negative prompts, run SAM mask proposal, accept, and refine masks with brush add/erase.
- SAM integration: supports auto-discovery of `sam3*.pt` / `sam3*.pth` files in the project root and remembers your last SAM path in project metadata.
- Dataset management: keeps originals in `images_all`, pose labels in `labels_all`, seg labels in `labels_seg_all`, and annotated overlays in `annotations`.
- Training launcher: choose task (pose, segmentation, or detection), dataset, epochs, batch size/device, and trigger `ultralytics.YOLO.train()` without leaving the UI.
- Video reviewer supports both workflows: pose overlays (bbox + keypoints) and segmentation overlays (mask polygons), with frame export back to `images_to_label`.
- Programmatic helper: `dataset_builder.py` generates YOLO pose `dataset.yaml` files with sensible flip indices based on keypoint names.

Release highlights (March 2026)
-------------------------------
- Added project launcher and project-scoped metadata/state management.
- Added segmentation workflow with SAM prompting, mask acceptance, and brush-based mask editing.
- Simplified segmentation UI to match pose UX patterns and cleaned dropdown readability.
- Added segmentation parity in Video Reviewer.
- Added automation tests for new studio workflow-aware logic.

Demo
---------------
- For a video walkthrough, see: [https://www.youtube.com/watch?v=aeKuTOTbb8c](https://youtu.be/aeKuTOTbb8c?si=wo0ss4DYnGRMW2T6)

Repository layout
-----------------
- `squeakpose_studio.py`: main PyQt6 application (project launcher, labeling, SAM seg tools, exporting, training, inference).
- `dataset_builder.py`: helper to emit YOLO pose `dataset.yaml` files.
- `images_to_label/`: drop raw frames to annotate (created on first run).
- `images_all/`, `labels_all/`, `labels_seg_all/`, `annotations/`: accumulated originals, pose labels, seg labels, and rendered overlays (auto-managed on Save).
- `datasets/`: created when exporting train/val splits.
- `runs/`: default Ultralytics training outputs.
- `classes.txt`, `keypoints.txt`: pose class and keypoint lists.
- `classes_seg.txt`: segmentation class list.
- `squeakpose_project.json`: project metadata (workflow + model/path state).
- `fonts/`: optional UI font (Fira Sans).
- `torch_ultralytics_checks.py`: quick check that PyTorch + MPS is working on Apple Silicon.

Requirements
------------
- Python >= 3.12 (3.12 Dev on Linux)
- PyQt6, torch, torchvision, ultralytics (declared in `pyproject.toml`)
- Optional: Apple Silicon / CUDA for faster training and inference (device auto-selected in app)

Setup
-----
1) Create a virtualenv and install:
```
pip install -e .
```
   If using `uv`, the PyTorch indexes in `pyproject.toml` are preconfigured.

2) Ensure `classes.txt` and `keypoints.txt` exist in the project root. On first launch the app will create defaults (`mouse` class; 6 keypoints).

3) If you will use segmentation, provide your own SAM weights file:
- Place `sam3.pt` in the project root (recommended), or
- Load any `sam3*.pt` / `sam3*.pth` manually from the UI.
- SAM weights are not bundled with this repository.

Running the app
---------------
```
uv run squeakpose_studio.py
```
On launch, you will be prompted to select a **project folder**. The app stores labeling, datasets, training runs, templates, and inference outputs inside that selected project.
- Startup now opens a launcher screen with **Open Project** and **Create Project**.
- Choosing **Create Project** immediately launches class/keypoint setup for the new project.
- By default, the launcher starts in your Documents projects folder (`Documents/SqueakPose Studio Projects`).
- In-app project commands are available under **File → Open Project…** and **File → Close Project**.
- Workflow selector supports **Pose** and **Segmentation (SAM)** and restores last-used workflow per project.
- Main controls are organized as hot-corner panels:
  - top-left: navigation + labeling
  - top-right: video tools
  - bottom-left: dataset + training
  - bottom-right: model + inference

Per-project structure (auto-created)
------------------------------------
- `images_to_label/`
- `images_all/`
- `labels_all/`
- `labels_seg_all/`
- `annotations/`
- `datasets/`
- `runs/`
- `inference outputs/`
- `templates/`
- `classes.txt`
- `keypoints.txt`
- `classes_seg.txt`
- `class_keypoints.json`
- `squeakpose_project.json`

Labeling workflow
-----------------
1) Add images to `images_to_label/`.
2) Select workflow from the **Workflow** dropdown in the top-left panel.

Pose workflow:
- Modes/shortcuts: pan/zoom (1), bbox (2), keypoint (3), predict (4).
- Draw one box for the active class, place class-specific keypoints in order, set visibility states.
- **Save** writes pose labels to `labels_all/<image>.txt`, copies source image to `images_all/`, and renders overlay to `annotations/`.

Segmentation workflow (SAM):
- On first switch, you are prompted to define segmentation classes (stored in `classes_seg.txt`).
- Modes/shortcuts: pan (1), segment prompt (2), edit mask brush (E).
- Prompt semantics: left-click positive, right-click negative.
- Run SAM (`G`) to generate preview, then **Accept** to commit per-class mask.
- Brush editing: left-drag add, right-drag erase, `,`/`.` to resize brush.
- **Save** writes segmentation labels to `labels_seg_all/<image>.txt`, copies source image to `images_all/`, and renders overlay to `annotations/`.

For both workflows:
- Browse with filters (`All`, `Labeled`, `Unlabeled`, `Archive`) and use **Complete → Next Unlabeled** to advance.

Dataset export & training
-------------------------
- **Export Dataset**:
  - Pose workflow exports from `labels_all` to `datasets/pose` (or `datasets/detect` for bbox-only cases).
  - Seg workflow exports from `labels_seg_all` to `datasets/segment`.
- **Train Model**: Pick task/dataset/model/device/epochs/batch size and launch Ultralytics training. Outputs land in `runs/`.

Video inference
---------------
- Load a trained model via **Load Model**.
- **Video Reviewer** adapts to active workflow:
  - Pose: bbox + keypoint overlays.
  - Segmentation: mask polygon overlays (bbox fallback when mask is unavailable).
- **Inference** prompts for a video, batch size, and runs YOLO on batches. Results are saved as CSV to the active project's `inference outputs/` folder.

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
- Device selection is automatic (CUDA → MPS → CPU). Run `python torch_ultralytics_checks.py` to verify.
- If OpenCV is missing, install `opencv-python` to enable video inference.
- For missing class/keypoint files, create or edit `classes.txt` and `keypoints.txt` at the project root.
- Segmentation requires your own SAM weights (`sam3.pt` or compatible `sam3*.pt` / `sam3*.pth`). These are not included in this repo.
- Place `sam3.pt` at project root for auto-load, or load manually from the Segmentation tools panel.
