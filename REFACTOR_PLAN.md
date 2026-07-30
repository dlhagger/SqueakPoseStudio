# SqueakPose Studio Refactor Plan

## Purpose

Refactor SqueakPose Studio before adding substantial new features so that:

- UI changes are localized and easier to review.
- project data remains safe during failures and cancellation;
- model-heavy operations remain isolated from the Qt event loop;
- pose and segmentation behavior stays consistent;
- new workflows can reuse project, worker, and dataset services; and
- the application remains runnable after every refactor phase.

This is an incremental restructuring plan, not a rewrite. Existing project
formats, command-line entry points, worker protocols, and user-visible behavior
should remain compatible unless a change is explicitly approved.

## Implementation Status

Completed foundations:

- canonical `ProjectPaths` and project structure creation;
- atomic project metadata store with corrupt-file recovery;
- shared worker event validation, configuration loading, and shutdown helpers;
- Qt-free annotation models and pose/segmentation documents;
- transactional annotation-save and dataset-export services;
- extracted application startup, project launcher, class manager, and video
  review view; and
- thin repository-level compatibility launcher with the implementation rooted
  under `squeakpose/ui/main_window.py`; and
- extracted annotation graphics, video reviewer, training dialog, and
  distillation dialog modules; and
- clean test-only dependency declarations for CI.

Still in progress:

- further decomposition of pose and segmentation interaction methods inside the
  main annotation window; and
- optional consolidation of dialog-specific process controllers beyond the
  shared protocol, shutdown, and cancellation helpers.

## Current Baseline

- The application is launched directly from `squeakpose_studio.py`.
- `squeakpose_studio.py` contains roughly 9,400 lines and currently owns:
  project setup, metadata, annotation models and graphics items, the main
  window, dataset and training dialogs, prediction process management, video
  review, distillation, and application startup.
- Qt-free data handling is already separated into modules such as
  `squeakpose_core.py`, `label_io.py`, `dataset_ops.py`, `prediction_ops.py`,
  and `inference_ops.py`.
- Training, prediction, inference, analysis, and video-review work already runs
  in child processes.
- The baseline test suite contains 102 passing tests.

## Non-Negotiable Guardrails

1. Run the full unit suite before and after every phase.
2. Preserve atomic saves, rollback, backups, quarantine behavior, and partial
   inference output.
3. Do not combine structural refactors with feature changes.
4. Prefer moving tested code with minimal edits before redesigning it.
5. Keep `squeakpose_studio.py` as a compatible launcher throughout the work.
6. Do not change project metadata or label formats without migration tests.
7. Keep workers independently testable without importing the full GUI.
8. Commit each extraction separately so it can be reviewed or reverted.

## Target Boundaries

The exact filenames may evolve, but responsibilities should converge on the
following structure:

```text
squeakpose_studio.py              Thin application launcher
squeakpose/
├── app.py                        QApplication setup and startup
├── project/
│   ├── paths.py                  Project path definitions and creation
│   ├── metadata.py               Metadata loading, migration, and preferences
│   └── health.py                 Project inspection and cleanup coordination
├── annotation/
│   ├── models.py                 Bounding boxes, keypoints, annotations
│   ├── pose.py                   Pose annotation state and serialization
│   ├── segmentation.py           SAM prompts, masks, polygons, and brush state
│   └── graphics.py               Reusable QGraphicsItem/QGraphicsView classes
├── workers/
│   ├── protocol.py               JSON event/request validation
│   └── process.py                Shared QProcess lifecycle management
├── ui/
│   ├── main_window.py            Main-window composition and navigation
│   ├── class_manager.py          Pose and segmentation class management
│   ├── video_reviewer.py         Video review dialog and frame export
│   ├── training_dialog.py        Training configuration and process UI
│   └── distillation_dialog.py    Dataset extraction and distillation UI
└── services/
    ├── annotation_save.py        Transactional image/label/overlay saving
    ├── dataset.py                Validation and export orchestration
    ├── prediction.py             Prediction coordination and application
    └── inference.py              Video inference coordination
```

Existing Qt-free modules can move under the package later, after the UI
extractions are stable. Moving them early would create import churn without
improving the highest-risk boundary.

## Phase 0 — Freeze and Characterize Behavior

Goal: make current behavior explicit before moving code.

Work:

- Record the current test command and supported Python version.
- Add tests for the highest-risk UI orchestration that is not yet covered:
  workflow switching, project switching, worker shutdown, prediction request
  correlation, and dataset-export replacement.
- Add a minimal offscreen startup test that constructs the main window against
  a temporary project.
- Capture a small set of representative UI screenshots using the existing
  screenshot helper.
- Document the JSON messages accepted and emitted by every worker.

Exit criteria:

- Existing 102 tests still pass.
- New characterization tests pass without real model weights.
- Worker event shapes are documented and covered by tests.
- A repeatable UI smoke check exists.

## Phase 1 — Extract Project Infrastructure

Goal: give every workflow one authoritative view of project paths and metadata.

Work:

- Move `_project_paths`, project directory creation, default-project handling,
  and project title logic into `squeakpose/project/paths.py`.
- Move project metadata read, migration, recovery, and preference persistence
  into `squeakpose/project/metadata.py`.
- Introduce a small immutable `ProjectPaths` data object while temporarily
  supporting dictionary-style callers through an adapter.
- Keep filesystem writes routed through the existing atomic helpers.

Tests:

- new-project structure;
- old metadata migration;
- corrupt metadata recovery;
- relative and absolute stored paths;
- unknown metadata fields surviving a save; and
- failures leaving existing metadata intact.

Exit criteria:

- UI code no longer constructs project subpaths ad hoc.
- Project metadata has one read/write implementation.
- Existing projects open without migration surprises.

## Phase 2 — Standardize Worker Communication

Goal: remove repeated and subtly different `QProcess` management from dialogs.

Work:

- Define the common worker event envelope in `workers/protocol.py`.
- Extract process start, stdout line buffering, stderr capture, cancellation,
  termination, temporary-config cleanup, and final-state handling into a
  reusable process controller.
- Preserve worker scripts as independently executable entry points.
- Add request identifiers everywhere a persistent worker can have more than one
  operation in flight.
- Normalize terminal outcomes to success, canceled, or failed.

Tests:

- split and partial JSON lines;
- malformed events;
- stale request identifiers;
- graceful cancellation followed by forced termination;
- process errors before startup;
- temporary-config cleanup; and
- exactly one terminal callback per run.

Exit criteria:

- Prediction, training, inference, analysis, review, and distillation dialogs
  use the shared lifecycle implementation.
- Closing any window reliably stops its owned worker.
- No worker-specific model logic moves back into the UI process.

## Phase 3 — Extract Annotation Domain and Graphics

Goal: separate annotation state from rendering and input gestures.

Work:

- Move `BoundingBox`, `Keypoint`, `KeypointEntry`, and `Annotation` into
  `annotation/models.py`.
- Move graphics items and view subclasses into `annotation/graphics.py`.
- Introduce explicit pose and segmentation document/state objects.
- Move label-to-domain and domain-to-label conversion behind those objects,
  while continuing to use `label_io.py`.
- Separate SAM prompt/mask state and mask editing from main-window state.
- Keep workflow-specific rules out of generic graphics items.

Tests:

- pose and segmentation state round trips;
- undo operations;
- per-class completeness;
- visibility handling;
- workflow switching without state leakage;
- polygon and brush transformations; and
- coordinate conversion at image boundaries.

Exit criteria:

- Annotation state can be tested without constructing the full main window.
- Graphics classes render and edit domain objects but do not perform project
  file I/O.
- Pose and segmentation logic no longer share mutable fields accidentally.

## Phase 4 — Extract Transactional Services

Goal: move multi-step operations out of event handlers without weakening data
integrity.

Work:

- Extract annotation saving into `services/annotation_save.py`.
- Extract dataset validation/export orchestration into `services/dataset.py`.
- Extract prediction application and inference launch configuration into their
  service modules.
- Return structured result objects rather than displaying message boxes inside
  services.
- Keep user prompts and presentation in Qt classes.

Tests:

- save rollback after each staged artifact;
- duplicate image stems;
- missing source images;
- canceled exports preserving the previous dataset;
- invalid prediction payloads;
- task mismatches; and
- structured error messages suitable for the UI.

Exit criteria:

- Main-window handlers mostly gather input, call a service, and present its
  result.
- Services have no dependency on `QMessageBox`.
- Existing transactional guarantees remain covered by failure-injection tests.

## Phase 5 — Split Major Dialogs and Main Window

Goal: reduce the main module to composition and cross-feature coordination.

Recommended extraction order:

1. Class-management dialogs.
2. Video reviewer.
3. Distillation dialog.
4. Training dialog.
5. Main-window annotation panels and controllers.
6. Application startup.

Each extraction should:

- move one coherent class or feature at a time;
- preserve public constructor behavior initially;
- avoid visual redesign;
- add focused tests before deleting the old implementation; and
- leave compatibility imports where needed for existing tests or scripts.

Exit criteria:

- `squeakpose_studio.py` is a thin compatibility launcher.
- No UI module owns unrelated project, annotation, and worker responsibilities.
- The main window composes smaller panels/controllers rather than implementing
  every workflow directly.
- Circular imports are absent.

## Phase 6 — Consolidate and Harden

Goal: remove transitional scaffolding and prepare stable extension points.

Work:

- Remove compatibility adapters that no longer have callers.
- Replace broad exception suppression with specific exceptions where practical,
  while retaining best-effort cleanup paths.
- Add type checking for Qt-free modules and worker protocols.
- Add formatting and linting with narrowly scoped rules.
- Review direct and transitive dependencies, especially OpenCV and analysis
  dependencies.
- Verify the CI test environment installs everything imported by the tests.
- Decide whether example datasets and model weights should use release assets,
  Git LFS, or a separate download.
- Update README architecture and contributor instructions.

Exit criteria:

- Unit, worker, startup, and UI smoke checks pass in CI.
- No unexpected warnings occur during a normal create/open/label/save flow.
- Dependencies needed at runtime are declared directly.
- Repository setup works from a clean clone.

## Feature-Readiness Gate

Large new features can begin once Phases 0 through 4 are complete. At that
point, new work should have stable interfaces for:

- project paths and metadata;
- annotation documents;
- transactional save/export services;
- worker requests and events; and
- process lifecycle and cancellation.

Phases 5 and 6 can continue alongside small, localized features if the feature
does not modify a component currently being extracted.

Before approving feature work, verify:

- all baseline and added tests pass;
- a temporary project can be created, reopened, and switched;
- both workflows can load, edit, save, and reload an annotation;
- dataset validation and all three export formats still work;
- workers cancel cleanly;
- prediction and video inference reject task-mismatched models; and
- no project-format migration is required by the refactor.

## Suggested Pull Request Sequence

Keep changes small enough to review independently:

1. Characterization tests and worker protocol documentation.
2. Project paths extraction.
3. Project metadata extraction.
4. Shared worker protocol.
5. Shared Qt process controller.
6. Annotation data models.
7. Graphics classes.
8. Pose state.
9. Segmentation state.
10. Annotation-save service.
11. Dataset orchestration service.
12. Class-management dialogs.
13. Video reviewer.
14. Distillation dialog.
15. Training dialog.
16. Main-window composition and launcher cleanup.
17. Dependency, typing, linting, and documentation hardening.

## Measures of Success

Line count is not the primary goal, but it is a useful warning signal.
Success should instead be measured by:

- a change to one workflow touching only its domain, service, and UI module;
- worker lifecycle fixes applying to every worker;
- project-format logic having a single owner;
- core annotation behavior being testable without the full GUI;
- fewer broad exception handlers in business logic;
- faster review of changes because responsibilities are clear; and
- no regression in project-data safety.
