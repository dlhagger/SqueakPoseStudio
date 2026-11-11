# SqueakPose Studio Architecture

SqueakPose Studio is being refactored incrementally around four boundaries. Existing
project layouts, annotation formats, worker entry points, and the
`squeakpose_studio.py` launcher remain compatibility contracts.

## Ownership

- `squeakpose.project` owns project paths, metadata, locking, recovery, safety,
  health checks, and session state.
- `squeakpose.annotation` owns typed pose and segmentation documents, Qt-free edit
  state, mask geometry, and assistant selection rules. Graphics modules translate
  that state into scene items but do not write project files.
- `squeakpose.services` owns workflow decisions: saving, dataset export, image queues,
  frame-annotation file coordination, prediction, inference, SAM requests, analysis,
  training, distillation, OpenAI authorization/request contracts, and video-review
  planning. Services return values or
  structured errors and never display dialogs.
- `squeakpose.workers` owns newline-delimited JSON parsing and process lifecycle.
- `squeakpose.ui` gathers user input, presents results, and coordinates the boundaries
  above. It is the only layer that should own Qt widgets and message boxes.
  `CanvasScenePresenter` constructs segmentation, prompt, and saved-reference scene
  items from already-loaded values; `DepthPreviewPresenter` renders validated depth
  artifacts and probe state. `dialog_launch` plans feature-dialog arguments without
  constructing a dialog or reaching through the main window.

Root-level modules are transitional compatibility surfaces. New domain behavior should
be placed under `squeakpose`; root imports can delegate until supported callers have
migrated.

## Import direction

The package dependency direction is intentionally one-way:

```text
squeakpose.core
      ↑
project / annotation state
      ↑
services / worker protocol
      ↑
UI and repository-level worker entry points
```

`squeakpose.core`, project state, annotation state, and services must not import
repository-root implementation modules. The supported root modules
(`squeakpose_core`, `dataset_builder`, `dataset_ops`, `depth_ops`, `label_io`,
`layer_ops`, `prediction_ops`, `inference_ops`, and `ui_style`) are compatibility shims
that point inward to canonical package implementations. Internal code must use the
package path. A focused AST regression test enforces this rule for annotation, project,
service, and worker packages.

Package initializers resolve convenience exports lazily. In particular, importing
`squeakpose.annotation.pose` or `squeakpose.annotation.segmentation` must not initialize
PyQt graphics; UI-only modules are loaded only when their exports are requested.

## Annotation instances

Pose and segmentation documents retain multiple instances per class. The existing
mapping API remains a view of the selected instance for each class, preserving
legacy controllers and single-animal callers. Whole-frame saves, dataset-facing
serialization, and reference rendering use `all_entries()` or typed snapshots.
`AnnotationEntries` snapshots carry the complete instance collection and selected
slots through existing frame-loading interfaces; casting one to a plain `dict`
loses that extra collection and must not be used for whole-frame replacement.

Explicit add/select operations park the previous instance rather than overwriting
it. Inactive instances are read-only scene references, excluded from existing
class-based scene queries. Editor instance IDs are not tracking identities and are
not written into YOLO rows. Loading assigns new local IDs while preserving row
order. Multi-animal prediction is opt-in; the default best-per-class planner remains
available to existing callers. Training/export formats and inference identity
contracts remain unchanged.

Segmentation-to-pose transfer uses shared image pixels. The frame service resolves
matching class schemas and exposes every saved mask's polygon bounds. The
`segmentation_transfer` service builds detached pose documents for creation or
box-only replacement. Replacement scores actual mask-polygon containment of
located keypoints (visible points weighted above inferred points), then box
geometry. A maximum-weight one-to-one assignment avoids mask reuse and greedy
pairing errors. The score gap to the next-best assignment and positive spatial
evidence determine whether the UI needs a chooser; ordinary separated objects
and overlapping boxes with distinct polygon/keypoint evidence apply automatically.
This is same-image spatial association, independent of instance numbers or labeling
order, not persistent identity mapping. Creating new boxes needs no association.
Replacing boxes preserves instance IDs and all
keypoint fields. Imported boxes with unplaced keypoints remain editable drafts;
completion and save checks evaluate every instance before writing training labels.

## Worker contracts

Protocol-aware workers write one JSON object per stdout line. Every protocol object has
a non-empty `event` field. Non-JSON stdout and stderr are diagnostic text and must not
be interpreted as state changes. Config-file reads are bounded, configs are
project-local temporary files, and the owning controller removes them on every terminal
path. The distillation CLI is the exception: it emits third-party/plain-text progress,
which the same controller deliberately treats as diagnostic output.

One-shot workers are owned by `WorkerJobController`. A job can emit progress and plain
output before terminating; the controller captures stderr, escalates cancel from
terminate to kill, and emits exactly one terminal result. Analysis, training,
distillation, video-review passes, and inference passes use this mode.

Cancel is cooperative first. `WorkerJobController.cancel()` atomically creates a
`<config_path>.cancel` marker next to the job's project-local config (best-effort,
only when the job has a config) and then calls `terminate()`; the marker is the
portable channel because `terminate()` does not reach console children on Windows.
Nothing is written to stdin: a worker thread blocked reading the stdin pipe makes
inherited-stdin grandchildren (Ultralytics runs `nvidia-smi`, `git`) hang during CRT
start-up on Windows. Workers poll for the marker with `start_cancel_file_watcher`
(daemon thread, every 0.25 s) and keep a cancel that arrives while the config loads;
workers launched without a config rely on signals only. The controller removes the
marker together with the config on every terminal path. The first SIGTERM/SIGINT
only sets the cancel flag; workers then stop at their own checkpoints and still emit
a `result` with `canceled: true`: inference and video review keep rows/predictions
already written; analysis raises `OperationCanceled` (a `BaseException`) at every
progress step and rendered frame, which unwinds through `AnalysisOutputTransaction`
so the previous published analysis is untouched; training sets Ultralytics'
`trainer.stop` (checked after every batch and after each epoch's validation), so the
current batch finishes and `last.pt`/`best.pt` are saved (a cancel before the first
batch aborts outright). Training reports `canceled: true` only when the stop actually
skipped work, and copies the full checkpoint to `weights/last_resumable.pt` (reported
as `resume_checkpoint`) before Ultralytics' final evaluation strips the optimizer
from `last.pt`; exact resume accepts that file and rejects finalized checkpoints.
Escalation is two-stage: a second signal raises `SystemExit(130)`, which still
unwinds transactions and `finally` blocks. `WorkerJobController.shutdown()` (window
close) creates the marker, calls `terminate()`, waits half the terminate timeout,
calls `terminate()` again, waits the rest, then kills; plain `shutdown_qprocess()`
for processes without a controller keeps a single terminate before the kill.
`force_stop()` kills immediately; the analysis and training dialogs expose it as a
second click on their cancel buttons.

Prediction and the interactive SAM assistant use separate `PersistentWorkerSession`
instances. Each worker emits `ready`, then accepts newline-delimited request objects on
stdin. Load and prediction replies carry the caller's `request_id`; their services
decide whether an event is current, stale, a background-load error, discardable because
the displayed image changed, or applicable. Both use `command: "shutdown"` for graceful
shutdown, followed by terminate/kill escalation when it does not complete.
`SamAssistantController` is the Qt-side lifecycle owner; the main window selects weight
paths, submits prompt values, and renders decisions without importing or constructing a
model.

The optional OpenAI pose assistant is remote I/O rather than local model execution.
`OpenAIController` performs browser OAuth, model discovery, and streaming Responses
requests on a daemon thread so Qt remains responsive. Its Qt-free services validate
OIDC tokens and structured model output. Credentials are installation-global,
owner-readable only, and never stored in a project; accepted proposals enter the
ordinary annotation document as separate editable instances and remain unsaved
until the user invokes Save. The structured response is an `instances` array with
one box and configured keypoint set per object, including an empty array when no
object is found. Parsing validates the entire response before application. After
confirmation, proposals replace only the requested class; other classes remain
intact. Responses for a changed image, layer, class, or keypoint schema are discarded.

The accepted event families are:

| Worker | Non-terminal events | Result/error contract |
| --- | --- | --- |
| Prediction | `ready`, `loading`, `loaded`, `started` | `result`, `error`, `stopped`; server replies are correlated by `request_id` |
| SAM assistant | `ready`, `loading`, `loaded`, `started` | `result`, `error`, `stopped`; one model remains warm and replies are correlated by `request_id` and image identity |
| Inference | `started`, `progress` | `result` includes output paths, row/frame counts, cancel/error state; `error` contains `error_message` |
| Video review | `started`, `batch_adjusted`, `progress` with bounded prediction chunks | `result` includes cancel/error state; partial chunks remain usable |
| Analysis | `started`, `progress` | `result` contains the analysis summary; `error` contains `error_message` |
| Training | `started`, `training` | `result` or `error` |

Fields used for coordination are part of the compatibility contract:

- Prediction server replies echo `request_id`. `started` also identifies `image_path`;
  `loading`/`loaded` identify `model_path`; `result` carries `canceled`, `had_error`,
  `error_message`, and `prediction`.
- SAM requests use `command: "load"` or `command: "predict"`, `request_id`,
  `model_path`, and optional `device`. SAM prediction requests additionally carry
  `image_path`, equal-length `points` and binary `labels`. Replies echo `request_id`;
  `loading`/`loaded` identify `model_path`, `started` identifies `image_path`, and a
  successful `result.prediction` carries contour `points`, `score`, and a stable
  failure value (`""`, `no_masks`, or `no_polygon`). Model construction and prediction
  run only in `sam_worker.py`; stdout from third-party code is redirected to stderr so
  protocol stdout remains JSON-only.
- Inference `progress` carries `processed_frames`, `total_frames`, and `message`.
  `result` carries `csv_path`, `preview_path`, `rows_written`, `processed_frames`,
  `canceled`, `had_error`, `error_message`, `mode`, and `layer_id`.
- Video-review `progress` carries `processed`, `total`, `effective_batch`, and a bounded
  `predictions` mapping. Its final `result` deliberately does not resend predictions;
  `preds_streamed` is true and `prediction_count` reports the completed total.
- Analysis progress uses `step`, `total`, and `message`. Training results use
  `canceled`, `had_error`, `error_message`, and `save_dir`.
- Every `error` event intended for presentation carries `error_message`. Consumers must
  tolerate extra fields and reject a missing/empty `event` envelope.

Unexpected or malformed protocol lines are reported as diagnostic output; they do not
silently mutate annotation state.

## Project-data rules

- Resolve paths through `ProjectPaths` and reject outputs that escape the project,
  including symlink escapes where a service accepts worker-reported paths.
- Stage multi-file writes and replace their targets only after every staged artifact
  succeeds.
- `squeakpose.services.frame_annotations` loads tolerant legacy pose/segmentation rows
  into typed documents and builds save requests from detached typed snapshots. It
  delegates row parsing/formatting to `squeakpose.annotation.serialization`; malformed
  row recovery and on-disk YOLO formats therefore remain compatibility behavior rather
  than a second codec.
- Preserve recovery, backup, quarantine, partial-inference, and project-lock behavior.
- A service may plan deletion, but the UI must present the scope before executing it.
- Annotation and cache snapshots returned across boundaries must be detached copies.

## Adding a workflow

1. Put data transitions and validation in a Qt-free domain or service module.
2. Characterize existing payloads and formats with focused tests.
3. Run model-heavy work in a child worker using the shared protocol.
4. Use `WorkerJobController` for a finite command or `PersistentWorkerSession` for a
   restartable request/response process.
5. Keep the UI handler limited to input collection, rendering, and user-facing errors.
6. Run Ruff, the full unit suite, and `git diff --check`.

## Verification

```bash
uv lock --check
uv run --locked --only-group dev ruff check .
uv run --locked --only-group dev ruff format --check .
uv run --locked --only-group dev mypy
uv run --locked --only-group test python -m unittest discover -s tests -q
QT_QPA_PLATFORM=offscreen uv run --locked --only-group test \
  python tests/render_ui_screenshots.py --output-dir /tmp/squeakpose-ui
```

The lifecycle suite sets `QT_QPA_PLATFORM=offscreen`, constructs a complete project and
`LabelingApp`, and verifies deterministic close and lock cleanup without loading model
weights. The screenshot smoke constructs the launcher, main workflow states, and large
dialogs. Both run from CI's locked `test` group rather than relying on the full runtime
environment.

Static type checking uses the pinned mypy version in the development group. Its initial
baseline covers `squeakpose/project`, `squeakpose/annotation`, `squeakpose/services`, and
`squeakpose/workers`. The configuration checks typed and untyped function bodies,
rejects implicit optional values, and reports redundant casts and unused suppressions.
Missing third-party stubs are ignored because the Qt and scientific/model stack does not
consistently publish typing metadata; `no_site_packages` keeps a full local environment
checking the same way as CI's dev-only install (PyQt6 6.12.0's bundled stubs do not parse); no local package is covered by a blanket
`ignore_errors` override.

## Dependency boundary audit

The 2026-08 Phase 6 audit retained all runtime requirements. Static import evidence from
the application and workers accounts for PyQt6, PyYAML, NumPy, OpenCV, pandas,
matplotlib, seaborn, scikit-learn, hdbscan, UMAP, one-euro-filter, PyAV (`av`), PyTorch,
Ultralytics, Lightly Train, HTTPX, and PyJWT. The notebook-facing IPython packages and
transitive or lower-level model-stack packages such as CLIP, timm, torchvision, tqdm,
and numba are
candidates for future optional feature groups, but moving them now could change
clean-install behavior for the bundled analysis notebook or distillation/model
workflows. They must be validated from clean feature-specific environments before any
removal from the runtime group.

The `test` group intentionally repeats its small scientific/Qt surface because CI uses
`uv sync --locked --only-group test`, which does not install the project runtime
dependencies. OpenCV and PyAV (`av`) are explicit in that group because tests import
annotation/analysis paths that load `cv2` and the video-encoder worker; relying on
either transitively made the reduced install incomplete. The `dev` group contains only
pinned repository tooling
(Ruff and mypy). A clean temporary workspace completed `uv lock --check`, locked
test-only sync, the complete unit suite, lifecycle smoke, and all screenshot renders.
A locked full dependency sync and the same lifecycle/render path also completed. No
runtime package was removed or reclassified during this audit.

## Mixed-class inference and segmentation analysis

Inference tracks every class without requiring a subject selection. Optional
count checks compare all confirmed tracks with an expected total; zero disables
them. Legacy subject metadata is still readable. Analysis worker configs carry
the class selected in the analysis dialog, independently of inference.

`services/class_tracking.py` wraps per-model tracker initialization with independent
ByteTrack/BoT-SORT instances per class. It remaps per-class detection indices back
to the original result indices so masks and keypoints remain paired correctly,
and emits a shared track-ID namespace within each model pass. No Ultralytics global
tracker registry is patched.

`services/analysis_subjects.py` applies the analysis class selection, with recorded
subject metadata as a legacy fallback. Segmentation
geometry exports preserve all detections; movement analysis splits subject data
by track and contiguous frame segments before smoothing and derivatives. Legacy
single-class inputs remain usable. Mixed-class inputs require an analysis class
choice rather than selecting an object implicitly.

All-class analysis uses temporary per-class computations inside one outer output
transaction. Only one class-labelled instance table is published; original track
IDs remain available and trajectory IDs are namespaced for unambiguous grouping.
One summary holds grouped class statistics, plots share the output `plots/`
directory, and one annotated video renders all instances. Intermediate class
CSVs and folders are removed with the temporary workspace. Failure in any class
leaves the previous published analysis intact.

Pose-only analysis splits detections by class, track, and contiguous frame
segments before smoothing and derivatives. Repeated same-class instances are
retained; untracked multi-instance rows receive no inferred temporal association.
Motion plots and ROI transitions respect trajectory boundaries. Reviewer cache
metadata includes an instance-format version so caches that may have collapsed
same-class detections cannot silently reappear as complete predictions.
