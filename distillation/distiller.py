#!/usr/bin/env python3
import argparse
import os
from typing import Any

DEFAULT_MODEL = "ultralytics/yolo26s-pose.pt"
DEFAULT_TEACHER = "dinov3/vitb16"
DEFAULT_PRECISION = "bf16-mixed"
DEFAULT_RUN_NAME = "dinov3-pose"


def default_data_dir(project_root: str) -> str:
    return os.path.abspath(os.path.join(project_root, "distillation", "unlabeled_images"))


def default_output_dir(project_root: str, run_name: str = DEFAULT_RUN_NAME) -> str:
    return os.path.abspath(os.path.join(project_root, "runs", "distillation", run_name))


def build_run_config(
    *,
    project_root: str,
    data_dir: str = "",
    out_dir: str = "",
    run_name: str = DEFAULT_RUN_NAME,
    model: str = DEFAULT_MODEL,
    teacher: str = DEFAULT_TEACHER,
    epochs: int = 300,
    batch_size: int = 64,
    precision: str = DEFAULT_PRECISION,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = os.path.abspath(project_root)
    resolved_data_dir = os.path.abspath(data_dir) if str(data_dir or "").strip() else default_data_dir(root)
    resolved_out_dir = os.path.abspath(out_dir) if str(out_dir or "").strip() else default_output_dir(root, run_name)
    return {
        "project_root": root,
        "data": resolved_data_dir,
        "out": resolved_out_dir,
        "model": model,
        "teacher": teacher,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "precision": str(precision),
        "overwrite": bool(overwrite),
        "run_name": str(run_name),
    }


def run_distillation(**kwargs) -> dict[str, Any]:
    config = build_run_config(**kwargs)
    data_dir = config["data"]
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Distillation image directory was not found: {data_dir}\n"
            "Create it manually or pass --data-dir to point at an existing image corpus."
        )
    out_dir = config["out"]
    out_parent = os.path.dirname(out_dir.rstrip(os.sep)) or out_dir
    os.makedirs(out_parent, exist_ok=True)

    import lightly_train

    lightly_train.pretrain(
        out=out_dir,
        overwrite=config["overwrite"],
        data=data_dir,
        model=config["model"],
        method="distillation",
        method_args={
            "teacher": config["teacher"],
        },
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        precision=config["precision"],
    )
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DINO distillation against a SqueakPose Studio project.",
    )
    parser.add_argument("--project-root", required=True, help="Path to the SqueakPose Studio project root.")
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
        default=DEFAULT_RUN_NAME,
        help="Name for the default distillation output folder under project runs.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Student model checkpoint or config to distill.")
    parser.add_argument("--teacher", default=DEFAULT_TEACHER, help="Teacher model identifier for Lightly Train.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of distillation epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for distillation.")
    parser.add_argument("--precision", default=DEFAULT_PRECISION, help="Lightning precision string.")
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        precision=args.precision,
        overwrite=args.overwrite,
    )
    print(f"Project root: {config['project_root']}")
    print(f"Data dir: {config['data']}")
    print(f"Output dir: {config['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
