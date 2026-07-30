"""Discovery helpers for project-local DINO distillation exports."""

from __future__ import annotations

import os


def distillation_export_search_roots(
    project_root: str,
) -> list[tuple[str, str]]:
    root = os.path.abspath(project_root)
    return [("Project runs", os.path.join(root, "runs", "distillation"))]


def preferred_distillation_export(run_dir: str) -> str:
    exported_dir = os.path.join(run_dir, "exported_models")
    if not os.path.isdir(exported_dir):
        return ""
    run_name = os.path.basename(os.path.abspath(run_dir.rstrip(os.sep)))
    preferred = [
        os.path.join(exported_dir, "exported_last.pt"),
        os.path.join(exported_dir, f"{run_name}_last.pt"),
    ]
    for candidate in preferred:
        if os.path.isfile(candidate):
            return candidate
    try:
        names = sorted(os.listdir(exported_dir))
    except OSError:
        return ""
    for name in names:
        candidate = os.path.join(exported_dir, name)
        if name.endswith(".pt") and os.path.isfile(candidate):
            return candidate
    return ""


def discover_distillation_exports(
    search_roots: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    exports: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for source_label, root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        run_dirs: list[str] = [root]
        try:
            for dirpath, dirnames, _ in os.walk(root):
                if "exported_models" in dirnames:
                    run_dirs.append(dirpath)
        except OSError:
            continue
        root_exports: list[tuple[str, str]] = []
        seen_run_dirs: set[str] = set()
        for run_dir in run_dirs:
            normalized_run_dir = os.path.abspath(run_dir)
            if normalized_run_dir in seen_run_dirs:
                continue
            seen_run_dirs.add(normalized_run_dir)
            checkpoint_path = preferred_distillation_export(normalized_run_dir)
            if not checkpoint_path:
                continue
            normalized_checkpoint = os.path.abspath(checkpoint_path)
            if normalized_checkpoint in seen_paths:
                continue
            seen_paths.add(normalized_checkpoint)
            run_label = os.path.relpath(normalized_run_dir, root)
            if run_label in {".", ""}:
                run_label = (
                    os.path.basename(normalized_run_dir)
                    or os.path.basename(root)
                )
            root_exports.append(
                (f"{source_label}: {run_label}", normalized_checkpoint)
            )
        root_exports.sort(key=lambda item: os.path.getmtime(item[1]), reverse=True)
        exports.extend(root_exports)
    return exports


def distillation_sample_count(
    total_frames: int,
    stride: int,
    max_frames: int = 0,
) -> int:
    total = max(0, int(total_frames))
    step = max(1, int(stride))
    count = (total + step - 1) // step
    if int(max_frames) > 0:
        count = min(count, int(max_frames))
    return count
