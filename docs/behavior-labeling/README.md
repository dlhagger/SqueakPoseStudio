# Behavior annotation in SqueakPose Studio

Watch a source video and mark the first and last frames of behaviors such as
rearing or freezing. The tool saves human annotations for later analysis and
model development. You do not need a trained model or inference results to label.

Only label the behaviors assigned to you. Everything else stays **unlabeled**;
unlabeled frames are not automatically examples of behavior absence.

## Install and launch

Install [uv](https://docs.astral.sh/uv/), then obtain the SqueakPose Studio
repository. If you use Git:

```bash
git clone https://github.com/dlhagger/SqueakPoseStudio.git
cd SqueakPoseStudio
```

From the repository folder:

```bash
uv sync --locked
uv run --locked python squeakpose_studio.py
```

The repository selects Python 3.14; uv can download it automatically. A desktop
environment is required. The installation includes the full application's
dependencies, even when you only use behavior annotation. No model weights or GPU
are required for this workflow.

For a coordinated annotation project, use the release or commit assigned by your
coordinator. Restart the application after updating its code.

## Prepare your project

1. At startup, choose **Create Project**, or **Open Project** to resume your work.
   Keep the project folder outside the application repository.
2. Open **Videos > Label Behaviors…**.
3. On first open, enter your target behavior names. Double-click a table cell to
   edit its name or definition; use **Add behavior** for additional rows.
4. Write the agreed criteria for what counts as each behavior, including when it
   starts and stops. Select **Save behaviors & continue**.
5. Choose a project video and select **Open Video**, or use **Browse…** to select
   a source recording.

Use the same behavior names and definitions as the other annotators on your
study. Each annotator should work in a separate project folder.

If a recording is stored in a cloud-synced folder, make sure the complete file is
available locally before opening it. Opening reads compressed video timestamps
to build a frame lookup; it does not decode every image before playback. The
lookup is held in memory and rebuilt when the video is reopened. Images are
decoded as needed. Opening time depends on the recording and storage.

## Label a behavior

1. Choose the **Animal** and **Behavior**. For a single animal, use **Animal 1**.
2. Play or seek to the first frame showing the behavior. Select **Mark Start**
   or press `I`.
3. Move to the last frame showing the behavior. Select **Mark End** or press `O`.
   Both selected frames are included.
4. Add an optional note and select **Add Interval**.

The saved interval appears in the timeline and table. Repeat for the next bout.
You do not need to label the gaps between bouts.

Marking start and end creates a draft. **Add Interval** saves it to the project;
an unsubmitted draft is not a saved annotation. The interface asks before
discarding unapplied changes.

### Playback and navigation

| Control | Action |
| --- | --- |
| `Space` | Play or pause |
| Left / right arrow | Step backward / forward one frame |
| `I` / `O` | Mark the first / last included frame |
| `1`–`9` | Select the corresponding behavior |
| **Go to frame** | Seek to a source frame number |
| Playback speed selector | Change viewing speed |
| **Zoom around playhead** | Zoom the behavior timeline near the current frame |

Shortcuts do not intercept typing in text or frame fields. Playback is silent.
For long recordings, zoom the timeline to see short bouts more clearly.

### Correct a saved interval

Click its table row or colored timeline segment. Adjust its behavior, frame
boundaries, or notes, then select **Update Interval**. You can also delete it,
drag a selected interval's timeline edges, or use **Split at Playhead**.
**New / Clear** returns to creating a new interval. **Undo** and **Redo** save
their changes too.

The current tool allows only one behavior per animal at a given frame.
Overlapping intervals for the same animal are rejected. Agree on how to handle
simultaneous behaviors before beginning a study.

### Edit the behavior list

Select **Behaviors…** in the top toolbar. Double-click names or definitions,
add or remove rows, and select **Save definitions**. Behaviors already used by
saved intervals in the open video cannot be removed through this editor.

The project list controls the choices for new labels, including on previously
annotated videos. Older intervals retain removed behaviors, shown as
**(saved label)** when editing those intervals. Older Other and Unscorable
annotations also remain readable and editable; neither category is added to the
choices for new labels.

Coordinate definition changes with your study team. Renaming a behavior changes
its meaning in the annotation record, so do not reuse a name or identifier for a
different behavior during collection.

## Save, export, and submit

Added or updated intervals autosave to your project:

```text
your-project/
└── behavior labels/
    ├── catalog.json       # Project behavior definitions
    └── <video_id>.json    # Video annotations and resume position
```

Reopening the same recognized recording restores its saved annotations and
playback position. Keep a backup of your project folder.

Use **Export JSON…** to submit the complete annotation record. **Export CSV…**
provides one row per interval for inspection and downstream analysis. Export to
a separate submission folder, outside the managed `behavior labels` directory.
Keep the original project files so you can resume editing.

For example, a study could collect:

```text
submissions/
└── annotator-A_recording-001/
    ├── annotations.json
    ├── annotations.csv
    └── submission.txt
```

Record the following in `submission.txt` or your study's assignment sheet:

- Annotator identifier.
- Assigned recording identifier and exact source filename.
- Behavior-definition version or the date the shared definitions were agreed.
- SqueakPose Studio release or Git commit used.
- Date and any notes about incomplete review or uncertain bouts.

Annotator identity and application version are not currently recorded
automatically in the annotation export. This submission convention is a manual
handoff, not a built-in upload or import feature.

## Frame alignment and current limits

- Frame numbers are **zero-based source frame indices**, intended to align with
  inference and per-frame analysis for the same source recording.
- The UI includes the last selected frame. Exports use an exclusive end:
  selecting frames 100 through 129 produces `start_frame = 100` and
  `end_frame_exclusive = 130`, covering 30 frames.
- Timestamps come from the video and supplement the frame boundaries. Do not
  replace saved frame numbers with estimates calculated from nominal FPS.
- Video identity currently includes sampled content, file size, and modification
  time. Copies with different modification times can receive different IDs.
  Use your study's shared recording identifier to reconcile submissions across
  computers; do not assume independently generated `video_id` values will match.
- Animal identities are assigned manually within each recording. They are not
  automatically linked to inference track IDs.
- The workspace shows the source video. It does not display inference overlays,
  train a behavior classifier, merge labels into inference CSVs, or import
  external annotation exports.
- Supported fast-indexing paths cover MP4/MOV, MKV/WebM, and AVI with supported
  codecs including H.264, HEVC, AV1, VP8/9, MJPEG, and FFV1. Unsupported or
  ambiguous frame layouts produce an error rather than guessed frame numbers.
  Keep the assigned source recording unchanged and report opening errors to the
  coordinator before substituting a re-encoded video.

See the [main README's file-format documentation](../../README.md#behavior-annotation-files)
for the versioned JSON and CSV fields.
