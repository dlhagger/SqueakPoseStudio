# SqueakPose Studio

SqueakPose Studio is a desktop application for small-animal image annotation,
YOLO dataset creation, model training, prediction review, and video inference.
It uses independent **Keypoints**, **Segmentation**, and **Depth** layers. The
editable layers keep separate labels, classes, dataset/training defaults, and
analysis context; every layer keeps its own model and inference outputs while
sharing the same project images.

The application is built with PyQt6, PyTorch, and Ultralytics YOLO. Model-heavy
operations run in child processes so training, prediction, and inference do not
block the main interface.

[![Tests](https://github.com/dlhagger/SqueakPoseStudio/actions/workflows/tests.yml/badge.svg)](https://github.com/dlhagger/SqueakPoseStudio/actions/workflows/tests.yml)

## Features

- Project launcher with **Create Project** and **Open Project** workflows.
- Pose annotation with per-class bounding boxes, ordered keypoints, templates,
  visibility states, and model-assisted predictions.
- Segmentation annotation with positive and negative SAM prompts, mask previews,
  and add/erase brush editing.
- Inference-only depth maps for images, with raw NumPy data, color previews,
  and compact metadata saved together.
- YOLO keypoint/pose and segmentation dataset export with reproducible
  train/validation splits.
- Layer-aware YOLOv26 training for keypoints/pose and segmentation.
- Support for standard YOLO models, existing checkpoints, exact run resume, and
  optional DINO-distilled backbones.
- Single-image prediction and a video reviewer for finding and exporting useful
  frames back into the labeling queue.
- Video inference with per-frame CSV output; depth passes also save a colorized
  MP4 preview.
- Qt-free helper modules with unit tests for label, dataset, prediction,
  inference, and training logic.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 or newer
- A desktop environment supported by PyQt6
- Optional CUDA or Apple Silicon acceleration
- Optional SAM-compatible weights for the Segmentation layer

PyTorch package sources are configured in `pyproject.toml`: macOS uses the
PyTorch CPU index, while Linux and Windows use the configured CUDA index. The
application selects CUDA, MPS, or CPU at runtime according to availability.

## Install and Run

From the repository root, create the environment and install the locked
dependencies:

```bash
uv sync
```

Launch the application:

```bash
uv run python squeakpose_studio.py
```

This is a dependency-only uv project; the repository itself is run directly and
is not installed as a Python package.

## Projects

At startup, choose **Create Project** or **Open Project**. By default, the
launcher starts in:

```text
~/Documents/SqueakPose Studio Projects
```

A project stores its own images, labels, datasets, training runs, templates,
inference outputs, and UI state. The active editing layer, the YOLO model
selected for each layer, and the most recent SAM weights path are restored from
project metadata.

New projects receive a default `mouse` pose class with six keypoints. Classes
and keypoints can be changed through the application before labeling begins.

### Project Structure

The application creates the following entries inside each project:

```text
project/
├── images_to_label/       # labeling queue
├── images_all/            # images that have been saved or validated
├── labels_all/            # YOLO pose labels
├── labels_seg_all/        # YOLO segmentation labels
├── annotations/
│   ├── keypoints/         # rendered keypoint previews
│   └── segmentation/      # rendered segmentation previews
├── depth maps/
│   ├── images/            # raw float32 .npy maps and JSON metadata
│   └── previews/          # colorized depth-map PNG files
├── datasets/              # exported YOLO datasets
├── runs/                  # training output
├── templates/             # pose annotation templates
├── inference outputs/
│   ├── keypoints/         # keypoint-layer video inference CSV files
│   ├── segmentation/      # segmentation-layer video inference CSV files
│   ├── depth/             # depth CSV summaries and preview videos
│   └── runs/              # manifests linking multi-layer inference passes
├── analysis outputs/
│   ├── keypoints/         # keypoint-layer analysis runs
│   ├── segmentation/      # segmentation-layer analysis runs
│   └── depth/             # reserved for future depth analysis
├── logs/
├── classes.txt
├── keypoints.txt
├── class_keypoints.json
├── classes_seg.txt
└── squeakpose_project.json
```

Use **File > Open Project...** to switch projects or **File > Close Project**
to return to the launcher.

Operational events and recoverable failures are written as structured JSON
lines to `logs/squeakpose.jsonl` inside the active project. The log rotates at
5 MiB and retains three backups. Opening a different project redirects new log
events to that project's log directory.

Projects use an ownership-token lock at `.squeakpose.lock` so two application
windows cannot write the same project concurrently. A lock is removed during
normal shutdown. If the owning process on the same computer is no longer
running, the application asks before removing the proven stale lock; invalid or
unverifiable locks are never removed automatically. Managed annotation,
dataset, metadata, and log paths are resolved through symlinks and rejected if
they escape the active project.

## Labeling

Add source images to the project's `images_to_label/` directory, then select
**Keypoints Layer**, **Segmentation Layer**, or **Depth Layer** from the layer selector.

Switching layers changes what is editable without replacing the other layer's
data. The active layer determines which label directory, class schema, YOLO
model, dataset export, training task, inference destination, and analysis
validation rules are used. Saved annotations from the other layer can remain
visible as dimmed, read-only references and can be toggled with the layer
visibility pills.

The **Keypoints**, **Segmentation**, and **Depth** pills are independent
visibility toggles. For example, while editing Keypoints you can switch the
saved Depth preview on as a translucent image overlay without changing the
active editing layer. Multiple non-active layers can be visible together, and
their visibility choices are saved with the project.

Image filenames must have unique stems, including across extensions and letter
case. For example, `frame.jpg`, `frame.png`, and `Frame.jpg` cannot coexist in
one labeling queue because each would map to the same YOLO label name. The
application excludes conflicting files until they are renamed.

The main window is organized into four control areas:

- Top left: image navigation and labeling controls
- Top right: video review
- Bottom left: dataset validation, export, and training
- Bottom right: model loading, prediction, and inference

The image browser can show all, labeled, or unlabeled images. Saving an image
writes its active-layer labels, copies the image into `images_all/`, and
renders an overlay into `annotations/`. These outputs are staged and committed
together, so a failed save does not replace an existing annotation.

### Keypoints Layer

For each active class:

1. Draw one bounding box.
2. Place the class's keypoints in their configured order.
3. Set keypoint visibility when a point is occluded or not visible.
4. Save the completed annotation.

Keypoint labels use the YOLO pose format and are written to
`labels_all/<image-stem>.txt`. Visibility values are:

- `2`: visible
- `1`: labeled but occluded
- `0`: not visible

Pose templates can be saved and applied per class. Once the project contains
labels, the existing class/keypoint schema is protected from removal, renaming,
or reordering because those changes would invalidate saved rows. New entries
can still be added.

### Segmentation Layer

Segmentation classes are stored in `classes_seg.txt`. SAM weights are not
included in this repository.

To use segmentation:

1. Put a compatible `sam3*.pt` or `sam3*.pth` file in the project root, or
   select one from the segmentation controls.
2. Use left-click prompts for foreground and right-click prompts for
   background.
3. Run SAM to generate a preview.
4. Accept the preview, then optionally refine it with the mask brush.
5. Save the completed masks.

The application automatically looks for SAM weights in the project root and
prioritizes `sam3.pt`. Accepted polygons are written in YOLO segmentation
format to `labels_seg_all/<image-stem>.txt`.

In mask edit mode, left-drag adds to a mask and right-drag erases from it.

### Depth Layer

The Depth layer is an inference-only, model-assisted labeling tool in the MVP,
similar to SAM for Segmentation. In the left-side **Depth Assistant** panel,
choose a compatible checkpoint or use the YOLO26 Depth menu to choose `n`,
`s`, `m`, `l`, or `x`. Built-in models are
downloaded by Ultralytics the first time they are used if not already cached;
larger variants trade additional compute for better benchmark accuracy.

Select an image and run **Predict**. A successful prediction saves:

- `depth maps/images/<image-stem>.npy`, the raw float32 depth map;
- `depth maps/images/<image-stem>_depth.json`, model and range metadata; and
- `depth maps/previews/<image-stem>_depth.png`, the colorized display image.

The displayed colors use inverse depth so nearer regions appear brighter. The
stored numeric values remain unchanged and are labeled `model_default`; the
MVP does not perform scene-specific scale calibration or depth-map editing.
The separate **Depth Display** panel switches between the original image,
standalone depth map, and a blended overlay. **Depth Range** reports the saved
2nd–98th percentile range and median in meters so unusually compressed or
extreme predictions are easier to spot.

While viewing the Depth layer, right-click any image pixel to sample its raw
`.npy` depth value. Numbered markers remain visible for up to six probes, and
the Depth Range panel lists their pixel coordinates, values in meters, and the
absolute difference between the two most recent valid samples. This supports
quick comparisons between an animal mask and nearby background; **Clear
Probes** removes the current markers.

When saved keypoints are toggled on in the Depth layer, each visible keypoint
label also includes the aligned raw depth (for example, `nose · 0.842 m`).
This is a display-only spot check and does not modify the keypoint annotations.

### Main Shortcuts

| Shortcut | Action |
| --- | --- |
| `1` | Pan/zoom mode |
| `2` | Bounding-box mode or segmentation prompt mode |
| `3` | Keypoint mode |
| `4` | Model prediction mode |
| `S` | Save |
| `Z` | Undo |
| `P` / `N` | Previous / next image |
| `K` | Skip to next unlabeled image |
| `Ctrl+Enter` / `Cmd+Enter` | Complete and open next unlabeled image |
| `Delete` / `Backspace` | Delete selected annotation |
| `Escape` | Cancel the current draw or SAM prompts |
| `Space` | Temporarily pan |
| `V` | Toggle selected keypoint visibility |
| `0` | Mark the next keypoint invisible |
| `A` | Apply the active pose template |
| `C` | Select the next class |
| `G` | Run SAM |
| `Shift+Enter` | Accept the SAM preview |
| `E` | Edit the accepted segmentation mask |
| `,` / `.` | Decrease / increase segmentation brush size |
| `R` | Reset zoom |

## Dataset Validation and Export

Use **Validate Labels** before export to normalize the active layer's label
files. Validation:

- creates a backup before the first rewrite;
- clamps normalized coordinates;
- pads missing pose keypoints as invisible;
- removes invalid rows;
- moves empty or completely unusable label files into a timestamped quarantine
  directory beside the active label directory; and
- ensures labeled images are available in `images_all/` when a matching queue
  image exists.

Use **Export Dataset** to choose a training ratio and deterministic shuffle
seed. Only images with usable labels for the selected layer are
included; images without matching labels are skipped and reported in the export
summary. Pose and segmentation rows are normalized again while the staged
dataset is built, so exported files match the current class/keypoint schema.
Exports are written under:

```text
datasets/pose/
datasets/segment/
```

The Keypoints layer exports YOLO pose labels, and the Segmentation layer exports
polygon labels. Each dataset contains YOLO `images/train`, `images/val`, `labels/train`,
`labels/val`, and `dataset.yaml` entries.

Pose dataset YAML files include class names, keypoint names, keypoint shape, and
flip indices inferred from `left` and `right` keypoint names.

Exports are built in a temporary directory first. Replacing an existing dataset
only occurs after all image/label copies and `dataset.yaml` generation succeed.
Canceling or failing an export leaves the previous dataset unchanged.

Use **Project Health** to inspect image counts, usable labels, orphan labels,
ambiguous stems, likely numbered image copies, worker config files, and stale
transaction artifacts. Its cleanup prompt removes only transaction staging
paths; it does not delete images, labels, worker configs, or transaction backups.
When a project opens, SqueakPose automatically restores a missing target only
when exactly one recognized transaction backup exists. Conflicting or ambiguous
backups are preserved for manual review, and recognized staging files are removed
only after confirmation.

## Training

Open **Train Model** after exporting a dataset. Training is locked to the active
layer so a segmentation dataset cannot accidentally be launched as pose
training, or vice versa. The dialog supports:

- YOLOv26 `n`, `s`, `m`, `l`, and `x` model sizes;
- standard YOLO initialization;
- task-compatible DINO distillation exports for Keypoints and Segmentation;
- fine-tuning from an existing YOLO checkpoint; and
- exact resume from a run containing `weights/last.pt`.

Training runs in a child process and streams output into the dialog. Results are
stored under the active project's `runs/train/<task>/` directory.
The dialog blocks dataset/task mismatches before launch, and each worker checks
the loaded model's Ultralytics task before prediction, review, inference, or
training begins.

Device selection prefers CUDA, then MPS, then CPU. CUDA uses automatic batch
sizing in the UI; MPS requires a positive manual batch size.

Use **Distillation** in the Dataset & Training panel to explicitly create an
unlabeled image corpus from project videos and launch DINO distillation. Images
default to `<project>/distillation/unlabeled_images/`, and outputs are stored
under `<project>/runs/distillation/<run-name>/`. The GUI requires confirmation
before extracting frames and supports sampling intervals and per-video limits.
Choose **Keypoints** or **Segmentation** before starting a run; the matching
YOLO pose or segmentation student is selected automatically, and the resulting
export appears only in the compatible layer's Train Model dialog. The
project-aware command-line entry point is `distillation/distiller.py`.

## Prediction and Video Review

Use **Project Models…** to assign `.pt`, `.yaml`, or `.onnx` prediction models
to the Keypoints and Segmentation layers. The selections are remembered with
the project. SAM and Depth use their dedicated model-assistant panels on the
left side of the labeling window.

Single-image prediction applies the best prediction for each class to the
current image. Pose predictions populate boxes and keypoints; segmentation
predictions populate boxes and masks; depth prediction saves and displays a
dense map.

The **Video Reviewer** supports `.mp4`, `.mov`, `.avi`, and `.mkv` files. It
can:

- run every configured project prediction model sequentially over the same
  selected frame range, with configurable stride, batch size, and confidence
  thresholds;
- cache predictions beside the video in `<video-name>.sqp_preds.json`;
- display Keypoints and Segmentation predictions together, with independent
  overlay visibility controls; and
- export the current, random, low-confidence, or high-confidence frames into
  the active project's labeling queue.

When both model passes are available, low/high-confidence export first asks
which prediction layer to rank, then asks for a class or balanced-by-class
selection within that layer.

Exported frame names include a short identifier derived from the source video
path, preventing same-named videos from being mistaken for one another.

Video Reviewer shortcuts include:

| Shortcut | Action |
| --- | --- |
| `Left` / `Right` | Previous / next frame |
| `E` | Export current frame |
| `Shift+E` | Export lowest-confidence frames |
| `Shift+H` | Export highest-confidence frames |
| `Shift+R` | Export random frames |
| `+` / `-` | Zoom in / out |
| `R` | Reset zoom |

## Video Inference

Video inference runs every configured project prediction model sequentially.
Each layer writes its own compatible CSV, and a run manifest links the passes.
Depth inference writes one summary row per frame and a colorized preview MP4.
If one pass fails, completed output from the other layers is retained.

Pose output includes frame/time metadata, class and confidence, bounding boxes,
tracking IDs when available, speed values, and absolute/normalized keypoint
coordinates with confidence. Segmentation output includes detection metadata,
boxes, and mask polygons. Frames without detections receive an explicit
no-detection row.

Closing a window with an active worker terminates the child process before the
window exits and removes its temporary config file. Canceled inference keeps
already-written CSV rows and reports the output as partial.

## Analysis

For the Keypoints and Segmentation layers, use **Analysis** in the right-hand
panel to run the repeatable parts of
`analysis_toolset/analysis_toolkit.ipynb` from a GUI. The dialog opens in the
active layer's context, prefers that layer's inference directory, and rejects a
CSV whose schema belongs to the other layer. It loads a source
frame when a video is available, lets you click two scale points, draw named
rectangular ROIs, and then runs the workflow on an inference CSV with the
selected smoothing and output options.

The analysis worker writes `analysis_features.csv`, `analysis_summary.json`,
ROI summaries, and optional plots into the active project's
`analysis outputs/` directory. Optional video-dependent outputs include
annotated videos and behavior-cluster clips when UMAP/HDBSCAN clustering is
enabled.

Segmentation inference CSVs are routed through a segmentation-specific workflow.
It parses `mask_polygon`, computes mask geometry, tracks the primary mask
centroid per frame, writes `segmentation_detections.csv`, and includes
segmentation coverage, mask area, trajectory, and ROI occupancy outputs.

## Repository Layout

```text
squeakpose_studio.py       Backward-compatible launcher and import facade
squeakpose/                Extracted application package
squeakpose/app.py          QApplication setup and startup
squeakpose/project/        Canonical project paths and metadata persistence
squeakpose/annotation/     Qt-free annotation state and reusable views
squeakpose/services/       Transactional annotation and dataset operations
squeakpose/workers/        Worker protocol and process lifecycle helpers
squeakpose/ui/             Main window and feature dialogs
squeakpose/ui/main_window.py  Main annotation window
squeakpose/ui/video_reviewer.py  Video review and frame export
squeakpose/ui/training_dialog.py  YOLO training configuration
squeakpose/ui/distillation_dialog.py  DINO corpus and training workflow
squeakpose_core.py         Qt-free project and layer compatibility logic
label_io.py                Label parsing and normalization
dataset_builder.py         YOLO pose dataset YAML generation
dataset_ops.py             Dataset validation and export helpers
prediction_ops.py          Prediction serialization helpers
inference_ops.py           Video inference row generation
predict_worker.py          Single-image prediction process
video_review_worker.py     Video review prediction process
inference_worker.py        Video inference process
analysis_ops.py            Inference CSV analysis workflow
analysis_worker.py         Analysis child process
analysis_dialog.py         PyQt6 analysis dialog
train_worker.py            Ultralytics training process
tests/                     Unit test suite
example_datasets/          Example pose images, labels, and overlays
analysis_toolset/          Example inference analysis notebook
distillation/              Project-aware DINO distillation entry point
```

`squeakpose_studio.py` remains the compatible direct launcher while UI
responsibilities are incrementally moved into the `squeakpose` package. Project
formats and worker entry points remain backward compatible during this
transition.

## Development Checks

Install the locked developer tools without the application dependencies:

```bash
uv sync --locked --only-group dev
```

Check lint and formatting:

```bash
uv run --locked --only-group dev ruff check .
uv run --locked --only-group dev ruff format --check .
```

Apply Ruff's safe lint fixes and formatter locally:

```bash
uv run --locked --only-group dev ruff check --fix .
uv run --locked --only-group dev ruff format .
```

Run the unit tests:

```bash
uv run python -m unittest -q
```

Run the PyTorch and Ultralytics environment check:

```bash
uv run python torch_ultralytics_checks.py
```

The environment check reports the selected PyTorch device and verifies that the
core model stack imports successfully.

## Benchmarking

Datasets for benchmarking in the SqueakPose Studio publication can be found at: 

https://zenodo.org/records/20629815

## Dependency Docs
https://docs.ultralytics.com/help/FAQ#faq

https://pytorch.org/get-started/locally/

https://doc.qt.io/qtforpython-6/

## License

See [`LICENSE`](LICENSE).
