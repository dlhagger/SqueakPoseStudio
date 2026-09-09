# Repository Audit Fix Plan

This plan converts the September 2026 repository audit into an implementation sequence. The
work preserves existing project formats and compatibility entry points while addressing the
confirmed correctness and reliability gaps first.

## Implementation Status

Completed in this pass:

- Phase 1: the CI baseline is clean after formatting the inference-review work.
- Phase 2: inference discovery is model-specific, source targets are snapshotted when runs are
  planned, and retargeted links no longer inherit stale results.
- Phase 3: inference-review scanning retains bounded ranking pools while keeping full scan counts.
- Phase 4: pose, segmentation, and combined analysis publish through a rollback-capable staging
  transaction; failed reruns preserve the last successful output.
- Phase 5: optional recent-project persistence and invalid analysis destinations fail safely.
- Phase 6 hygiene: `/weights/` is ignored so local checkpoints are not accidentally committed.

Deferred follow-ups:

- Broader Bugbear/complexity rules, coverage thresholds, expanded mypy scope, and decomposition of
  the main window should land incrementally to avoid mixing large mechanical refactors with the
  correctness fixes above.
- The tracked demo CSV and notebook remain unchanged because moving or deleting published example
  data needs an explicit distribution decision (Git LFS, release asset, or reduced fixture).

## Goals

- Keep inference outputs associated with the exact video, layer, and model that produced them.
- Keep the last successful analysis intact until a replacement run completes successfully.
- Bound inference-review memory use for long videos.
- Turn expected filesystem failures into actionable UI errors instead of uncaught exceptions.
- Keep the working tree and CI quality gates clean without committing local model weights.

## Phase 1: Restore the CI Baseline

- Format the new inference-review service, dialog, and tests with Ruff.
- Confirm the lockfile is current.
- Run configured Ruff linting, formatting, mypy, the complete unit suite, and the offscreen UI
  smoke renderer.

Acceptance criteria:

- Every configured CI command exits successfully.
- Existing user changes remain intact.

## Phase 2: Correct Inference Source Identity

- Discover successful inference passes by video identity, layer, and model identity instead of
  selecting only the newest output for each video before applying the model filter.
- Select the newest run inside each `(video, layer, model)` group.
- Include model/run identity in review-candidate keys so candidates from different models do not
  overwrite each other.
- Preserve project-move compatibility while refusing filename-only fallback when the recorded
  source still exists and differs from the current video target.
- Add regression tests for:
  - multiple models run against the same video;
  - a project video link retargeted to a different source with the same filename;
  - relocated projects with stale absolute paths;
  - candidate-key separation across models.

Acceptance criteria:

- Selecting a model exposes that model's newest available output for every applicable video.
- Retargeted videos are never paired with inference generated from the old target.
- Existing moved-project behavior remains supported.

## Phase 3: Bound Inference Review Memory

- Finalize candidates one frame at a time while reading the CSV.
- Maintain bounded ranking pools and aggregate counters without retaining every frame candidate.
- Preserve track-transition checks, cancellation behavior, ranking semantics, and overlay data for
  retained candidates.
- Add a large synthetic scan regression that verifies retained candidate growth is bounded.

Acceptance criteria:

- Retained candidates scale with the configured pool size and ranking modes, not video duration.
- Existing review results and UI filters remain compatible.

## Phase 4: Make Analysis Publication Transactional

- Produce generated analysis artifacts in a sibling staging directory.
- Publish app-owned files and directories only after the full workflow succeeds.
- Preserve unrelated user files in stable analysis directories.
- Remove staging output after failures and cancellation while leaving the previous successful run
  untouched.
- Cover pose, segmentation, and combined workflows with injected-failure regression tests.

Acceptance criteria:

- A failed rerun leaves all prior successful generated artifacts unchanged.
- A successful rerun removes stale app-owned artifacts and installs the new result.
- User-created files in the output directory are never removed.

## Phase 5: Harden Expected Error Paths

- Treat last-project persistence as best-effort so an unwritable user-state location does not
  abort application startup.
- Move analysis output-directory creation inside the dialog's existing error boundary.
- Present or log concise, actionable failures and release project resources deterministically.
- Add focused tests for unwritable or invalid destinations.

Acceptance criteria:

- A valid project can open even if recent-project state cannot be saved.
- Analysis launch failures are shown in the dialog and do not escape the Qt callback.

## Phase 6: Strengthen Maintainability and Repository Hygiene

- Ignore local checkpoint directories such as `/weights/`.
- Add selected Ruff Bugbear checks after fixing or explicitly documenting existing findings.
- Establish coverage reporting, then choose a non-regressive threshold from the measured baseline.
- Expand mypy incrementally into workers, controllers, and extracted UI coordination modules.
- Continue decomposing `squeakpose/ui/main_window.py` along project lifecycle, layer switching,
  annotation persistence, and inference orchestration boundaries.
- Decide whether large demonstration data belongs in Git LFS, a release asset, or a reduced fixture.

Acceptance criteria:

- Local weights no longer appear as untracked repository content.
- New quality gates are reproducible locally and in CI.
- Large-file handling is documented and intentional.

## Verification

Run from the repository root:

```bash
uv lock --check
uv run --locked --only-group dev ruff check .
uv run --locked --only-group dev ruff format --check .
uv run --locked --only-group dev mypy
QT_QPA_PLATFORM=offscreen uv run --locked --only-group test \
  python -m unittest discover -s tests -q
QT_QPA_PLATFORM=offscreen uv run --locked --only-group test \
  python tests/render_ui_screenshots.py --output-dir /tmp/squeakpose-ui
git diff --check
```

## Delivery Order

1. CI baseline and inference-review fixes.
2. Transactional analysis publication.
3. Startup and filesystem error handling.
4. Quality-gate and repository-hygiene improvements.
5. Larger architectural decomposition as follow-up work after the functional changes stabilize.
