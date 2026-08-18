"""Qt-free validation and worker configuration for YOLO training."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import yaml

from squeakpose.core import infer_dataset_task
from squeakpose.project.layers import normalize_layer_id


class TrainingConfigError(ValueError):
    """A user-correctable training configuration error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TrainingConsoleBuffer:
    """Reduce terminal carriage-return animation to stable, readable log lines."""

    ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self) -> None:
        self._current = ""

    @property
    def pending(self) -> str:
        return self._current

    def feed(self, text: str) -> list[str]:
        """Consume a terminal chunk and return only newline-committed lines."""
        cleaned = self.ANSI_ESCAPE_RE.sub("", str(text or "")).replace("\x1b", "")
        completed: list[str] = []
        for character in cleaned:
            if character == "\r":
                self._current = ""
            elif character == "\n":
                if self._current.strip():
                    completed.append(self._current.rstrip())
                self._current = ""
            elif character == "\b":
                self._current = self._current[:-1]
            else:
                self._current += character
        return completed

    def finish(self) -> list[str]:
        """Return a final unterminated line and reset the buffer."""
        final = self._current.rstrip()
        self._current = ""
        return [final] if final.strip() else []


@dataclass(frozen=True, slots=True)
class TrainingWorkerConfig:
    layer_id: str
    model_cfg: str
    params: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "model_cfg": self.model_cfg,
            "params": dict(self.params),
        }


@dataclass(frozen=True, slots=True)
class TrainingRunPlan:
    """Validated model, dataset, and worker parameters for one training run."""

    source_mode: str
    dataset_yaml: str | None
    task: str | None
    model_cfg: str
    params: dict[str, Any]
    model_notice: str | None = None


def training_run_name(model_spec: str) -> str:
    """Return the stable filesystem-safe run name used by the training dialog."""

    value = str(model_spec or "")
    if value.lower().endswith((".pt", ".pth", ".yaml", ".yml")):
        value = os.path.splitext(os.path.basename(value))[0]
    else:
        value = os.path.basename(value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "model"


def normalize_training_run_label(label: str) -> str:
    """Return one portable, project-contained folder name for a user run label."""
    value = str(label or "").strip()
    value = re.sub(r"[\\/]+", "_", value)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:80].rstrip("._-")


def resolve_dataset_yaml(path: str) -> str:
    """Resolve a dataset root or YAML file to an existing dataset YAML path."""
    candidate = str(path or "").strip()
    if not candidate:
        raise TrainingConfigError("required", "Select a dataset folder before starting training.")
    if os.path.isdir(candidate):
        dataset_yaml = os.path.join(candidate, "dataset.yaml")
        if os.path.isfile(dataset_yaml):
            return dataset_yaml
        raise TrainingConfigError(
            "yaml_missing",
            "Could not find dataset.yaml in the selected folder.\n"
            "Select the dataset root (contains dataset.yaml) or the YAML file directly.",
        )
    if candidate.lower().endswith((".yaml", ".yml")) and os.path.isfile(candidate):
        return candidate
    raise TrainingConfigError("not_found", f"Path not found:\n{candidate}")


def infer_training_task_from_yaml(yaml_path: str) -> str | None:
    """Read a dataset task without allowing parse failures to escape into the UI."""
    try:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None
    return infer_dataset_task(payload)


def resolve_model_config(base_cfg: str, task: str | None) -> tuple[str, str | None]:
    """Select the YOLO model variant matching the requested dataset task."""
    task_value = str(task or "").strip().lower()
    if not task_value:
        return base_cfg, None

    has_yaml_ext = base_cfg.lower().endswith(".yaml")
    stem = base_cfg[:-5] if has_yaml_ext else base_cfg
    stem_clean = re.sub(r"-(pose|seg)$", "", stem, flags=re.IGNORECASE)
    notice: str | None = None
    if task_value == "pose":
        target = f"{stem_clean}-pose"
        resolved = f"{target}.yaml" if has_yaml_ext else target
        if resolved != base_cfg:
            notice = "Pose task detected → switched to pose variant of the model config."
    elif task_value == "segment":
        target = f"{stem_clean}-seg"
        resolved = f"{target}.yaml" if has_yaml_ext else target
        if resolved != base_cfg:
            notice = (
                "Segmentation task detected → switched to segmentation variant of the model config."
            )
    elif task_value == "detect":
        resolved = f"{stem_clean}.yaml" if has_yaml_ext else stem_clean
        if resolved != base_cfg:
            notice = "Detection task selected → using detection variant of the model config."
    else:
        resolved = base_cfg
    return resolved, notice


def build_training_worker_config(
    *,
    layer_id: str,
    model_cfg: str,
    params: Mapping[str, Any],
) -> TrainingWorkerConfig:
    """Build the existing training worker payload after validating required fields."""
    normalized_layer = normalize_layer_id(layer_id, default="")
    if not normalized_layer:
        raise TrainingConfigError("layer", f"Unsupported training layer: {layer_id!r}")
    model = str(model_cfg or "").strip()
    if not model:
        raise TrainingConfigError("model", "A model configuration or checkpoint is required.")
    if not isinstance(params, Mapping):
        raise TrainingConfigError("params", "Training parameters must be a mapping.")
    return TrainingWorkerConfig(normalized_layer, model, dict(params))


def build_training_run_plan(
    *,
    source_mode: str,
    dataset_path: str,
    base_model_cfg: str,
    checkpoint_path: str = "",
    selected_task: str | None = None,
    default_task: str | None = None,
    layer_task: str | None = None,
    device: str,
    epochs: int,
    batch: int,
    project_runs_dir: str,
    run_name: str = "",
) -> TrainingRunPlan:
    """Resolve all non-visual training choices into the existing worker parameters.

    ``source_mode`` is one of ``scratch``, ``dino``, ``checkpoint``, or
    ``resume``. The function performs no writes, which keeps it usable in tests
    and leaves directory creation and error presentation to the dialog.
    """

    mode = str(source_mode or "scratch").strip().lower()
    if mode not in {"scratch", "dino", "checkpoint", "resume"}:
        raise TrainingConfigError("source", f"Unsupported training source: {source_mode!r}")

    exact_resume = mode == "resume"
    dataset_yaml = None if exact_resume else resolve_dataset_yaml(dataset_path)
    checkpoint = str(checkpoint_path or "").strip()
    if mode in {"dino", "checkpoint", "resume"}:
        if not checkpoint or not os.path.isfile(checkpoint):
            message = (
                "Select a valid DINO distillation export (.pt) before training."
                if mode == "dino"
                else "Select a valid YOLO checkpoint (.pt) before continuing."
            )
            raise TrainingConfigError(
                "checkpoint_required",
                message,
            )
        if exact_resume and os.path.basename(checkpoint).lower() != "last.pt":
            raise TrainingConfigError(
                "resume_checkpoint",
                "For exact run continuation, select a weights/last.pt checkpoint.",
            )

    if not exact_resume and str(device).lower() == "mps" and int(batch) <= 0:
        raise TrainingConfigError(
            "mps_batch",
            "Automatic batch sizing is unavailable on Apple MPS.\n"
            "Set a positive batch size before starting training.",
        )

    requested_task = str(selected_task or "auto").strip().lower()
    dataset_task = infer_training_task_from_yaml(dataset_yaml) if dataset_yaml else None
    if mode == "dino":
        task = str(layer_task or "").strip().lower() or None
    elif exact_resume:
        task = None
    elif requested_task in {"pose", "detect", "segment"}:
        task = requested_task
    elif dataset_task in {"pose", "detect", "segment"}:
        task = dataset_task
    elif str(default_task or "").strip().lower() in {"pose", "detect", "segment"}:
        task = str(default_task).strip().lower()
    else:
        task = None

    if task and dataset_task and task != dataset_task:
        raise TrainingConfigError(
            "task_mismatch",
            f"The selected dataset is '{dataset_task}', but the training task is '{task}'.\n\n"
            "Choose the matching task or select a different dataset.",
        )

    model_cfg = checkpoint if mode != "scratch" else str(base_model_cfg or "").strip()
    notice = None
    if mode == "scratch":
        model_cfg, notice = resolve_model_config(model_cfg, task)
    if not model_cfg:
        raise TrainingConfigError("model", "A model configuration or checkpoint is required.")

    if exact_resume:
        params: dict[str, Any] = {"resume": True, "device": device}
    else:
        task_folder = (
            task
            if task in {"pose", "detect", "segment"}
            else ("pose" if mode == "dino" else "auto")
        )
        project_dir = os.path.join(os.path.abspath(project_runs_dir), "train", task_folder)
        if mode == "checkpoint":
            checkpoint_run = os.path.basename(os.path.dirname(os.path.dirname(model_cfg)))
            default_run_name = training_run_name(checkpoint_run or model_cfg)
            if not default_run_name.endswith("_continue"):
                default_run_name = f"{default_run_name}_continue"
        else:
            default_run_name = training_run_name(model_cfg)
        requested_run_name = str(run_name or "").strip()
        effective_run_name = normalize_training_run_label(requested_run_name)
        if requested_run_name and not effective_run_name:
            raise TrainingConfigError(
                "run_name",
                "Run name must contain at least one letter or number.",
            )
        effective_run_name = effective_run_name or default_run_name
        params = {
            "data": dataset_yaml,
            "epochs": int(epochs),
            "device": device,
            "exist_ok": False,
            "batch": -1 if int(batch) <= 0 else int(batch),
            "project": project_dir,
            "name": effective_run_name,
        }
        if task:
            params["task"] = task

    return TrainingRunPlan(
        source_mode=mode,
        dataset_yaml=dataset_yaml,
        task=task,
        model_cfg=model_cfg,
        params=params,
        model_notice=notice,
    )


__all__ = [
    "TrainingConsoleBuffer",
    "TrainingConfigError",
    "TrainingWorkerConfig",
    "TrainingRunPlan",
    "build_training_run_plan",
    "build_training_worker_config",
    "infer_training_task_from_yaml",
    "normalize_training_run_label",
    "resolve_dataset_yaml",
    "resolve_model_config",
    "training_run_name",
]
