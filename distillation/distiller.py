#!/usr/bin/env python3
import argparse
import json
import os
import uuid
from typing import Any

DISTILLATION_MANIFEST_FILENAME = "squeakpose_distillation.json"
DEFAULT_MODELS = {
    "pose": "ultralytics/yolo26n-pose.pt",
    "segment": "ultralytics/yolo26n-seg.pt",
}
DEFAULT_RUN_NAMES = {
    "pose": "dinov3-pose",
    "segment": "dinov3-segmentation",
}
DEFAULT_MODEL = DEFAULT_MODELS["pose"]
DEFAULT_TEACHER = "dinov3/vitb16"
DEFAULT_PRECISION = "auto"
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES["pose"]


def normalize_task(task: str) -> str:
    normalized = str(task or "").strip().lower()
    aliases = {
        "keypoint": "pose",
        "keypoints": "pose",
        "pose": "pose",
        "seg": "segment",
        "segmentation": "segment",
        "segment": "segment",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported distillation task: {task!r}")
    return resolved


def default_data_dir(project_root: str) -> str:
    return os.path.abspath(os.path.join(project_root, "distillation", "unlabeled_images"))


def default_output_dir(project_root: str, run_name: str = DEFAULT_RUN_NAME) -> str:
    return os.path.abspath(os.path.join(project_root, "runs", "distillation", run_name))


def build_run_config(
    *,
    project_root: str,
    data_dir: str = "",
    out_dir: str = "",
    run_name: str = "",
    model: str = "",
    teacher: str = DEFAULT_TEACHER,
    task: str = "pose",
    epochs: int = 300,
    batch_size: int = 64,
    precision: str = DEFAULT_PRECISION,
    overwrite: bool = False,
    resume_interrupted: bool = False,
    gradient_accumulation_steps: int = 2,
) -> dict[str, Any]:
    root = os.path.abspath(project_root)
    resolved_task = normalize_task(task)
    resolved_run_name = str(run_name or "").strip() or DEFAULT_RUN_NAMES[resolved_task]
    resolved_model = str(model or "").strip() or DEFAULT_MODELS[resolved_task]
    if os.path.isfile(resolved_model):
        resolved_model = os.path.abspath(resolved_model)
    if min(int(epochs), int(batch_size), int(gradient_accumulation_steps)) < 1:
        raise ValueError("Epochs, batch size, and accumulation steps must be positive.")
    resolved_data_dir = (
        os.path.abspath(data_dir) if str(data_dir or "").strip() else default_data_dir(root)
    )
    resolved_out_dir = (
        os.path.abspath(out_dir)
        if str(out_dir or "").strip()
        else default_output_dir(root, resolved_run_name)
    )
    return {
        "project_root": root,
        "data": resolved_data_dir,
        "out": resolved_out_dir,
        "model": resolved_model,
        "teacher": teacher,
        "task": resolved_task,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "precision": str(precision),
        "overwrite": bool(overwrite),
        "resume_interrupted": bool(resume_interrupted),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "run_name": resolved_run_name,
    }


def write_run_manifest(config: dict[str, Any], *, pending: bool = False) -> str:
    out_dir = os.path.abspath(config["out"])
    os.makedirs(os.path.dirname(out_dir) if pending else out_dir, exist_ok=True)
    manifest_path = (
        out_dir + ".squeakpose.json"
        if pending
        else os.path.join(out_dir, DISTILLATION_MANIFEST_FILENAME)
    )
    payload = {
        "schema_version": 2,
        "task": config["task"],
        "student_model": config["model"],
        "teacher_model": config["teacher"],
        "run_name": config["run_name"],
        "config": config,
        "status": config.get("status", "running"),
    }
    temp_path = f"{manifest_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(temp_path, manifest_path)
    return manifest_path


def run_distillation(**kwargs) -> dict[str, Any]:
    config = build_run_config(**kwargs)
    data_dir = config["data"]
    if not config["resume_interrupted"] and not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Distillation image directory was not found: {data_dir}\n"
            "Create it manually or pass --data-dir to point at an existing image corpus."
        )
    out_dir = config["out"]
    out_parent = os.path.dirname(out_dir.rstrip(os.sep)) or out_dir
    os.makedirs(out_parent, exist_ok=True)

    import lightly_train

    if config["resume_interrupted"] and config["overwrite"]:
        raise ValueError("Resume and overwrite cannot be used together.")
    pending_path = out_dir + ".squeakpose.json"
    # Lightning may relaunch this script for additional GPU ranks. Only the
    # original process owns run metadata and backup operations.
    distributed_child = int(os.environ.get("LOCAL_RANK", "0")) > 0
    if distributed_child:
        with open(pending_path, encoding="utf-8") as fh:
            config = json.load(fh)["config"]
        data_dir = config["data"]
    elif config["resume_interrupted"]:
        if not os.path.isfile(os.path.join(out_dir, "checkpoints", "last.ckpt")):
            raise ValueError("No checkpoints/last.ckpt exists to resume.")
        manifest_path = (
            pending_path
            if os.path.isfile(pending_path)
            else os.path.join(out_dir, DISTILLATION_MANIFEST_FILENAME)
        )
        with open(manifest_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        if "config" not in saved:
            raise ValueError("This older run has no saved configuration; use a new run name.")
        if saved.get("status") == "complete":
            raise ValueError("This run is already complete; use a new run name.")
        config = dict(saved["config"], out=out_dir, resume_interrupted=True, overwrite=False)
        data_dir = config["data"]
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Original image corpus is missing: {data_dir}")
    elif os.path.exists(out_dir) or os.path.exists(pending_path):
        if not config["overwrite"]:
            raise FileExistsError(f"Run already exists: {out_dir}")
        backup = out_dir + ".backup-" + uuid.uuid4().hex[:12]
        if os.path.exists(out_dir):
            os.rename(out_dir, backup)
        if os.path.exists(pending_path):
            os.rename(pending_path, backup + ".squeakpose.json")
        print(f"Previous run preserved at: {backup}", flush=True)

    model = config["model"]
    if os.path.isfile(model):
        from ultralytics import YOLO

        model = YOLO(os.path.abspath(model))
        if model.task != config["task"]:
            raise ValueError(f"Student task {model.task!r} does not match {config['task']!r}.")

    config["status"] = "running"
    # Keep startup metadata beside the run: Lightly requires an empty output
    # directory. Discovery also reads this file after a hard process kill.
    if not distributed_child:
        write_run_manifest(config, pending=True)
    try:
        lightly_train.pretrain(
            out=out_dir,
            overwrite=distributed_child,
            resume_interrupted=config["resume_interrupted"],
            data=data_dir,
            model=model,
            method="distillation",
            method_args={
                "teacher": config["teacher"],
            },
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            precision=config["precision"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
        )
    except BaseException:
        config["status"] = "interrupted"
        if not distributed_child:
            write_run_manifest(config, pending=True)
        raise
    if distributed_child:
        return config
    config["status"] = "complete"
    write_run_manifest(config)
    os.remove(pending_path)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DINO distillation against a SqueakPose Studio project.",
    )
    parser.add_argument(
        "--project-root", required=True, help="Path to the SqueakPose Studio project root."
    )
    parser.add_argument(
        "--data-dir",
        default="",
        help="Optional path to an unlabeled image corpus. Defaults to <project>/distillation/unlabeled_images.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory. Defaults to <project>/runs/distillation/<run-name>.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Name for the output folder under project runs. Defaults to a task-specific name.",
    )
    parser.add_argument(
        "--task",
        choices=("pose", "segment"),
        default="pose",
        help="Task-specific YOLO student head to preserve in the distilled export.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Student model checkpoint or config. Defaults to the YOLOv26n model for the selected task.",
    )
    parser.add_argument(
        "--teacher", default=DEFAULT_TEACHER, help="Teacher model identifier for Lightly Train."
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of distillation epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for distillation.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--resume-interrupted", action="store_true")
    parser.add_argument(
        "--precision", default=DEFAULT_PRECISION, help="Lightning precision string."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow Lightly Train to overwrite an existing output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = run_distillation(
        project_root=args.project_root,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        model=args.model,
        teacher=args.teacher,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        precision=args.precision,
        overwrite=args.overwrite,
        resume_interrupted=args.resume_interrupted,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    print(f"Project root: {config['project_root']}")
    print(f"Data dir: {config['data']}")
    print(f"Output dir: {config['out']}")
    print(f"Task: {config['task']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
