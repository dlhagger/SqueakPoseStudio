"""Subject-class selection shared by tracked analysis workflows."""

from __future__ import annotations

import pandas as pd

from squeakpose.services.analysis_errors import AnalysisError


def select_subject_rows(raw: pd.DataFrame, subject_class: str = "") -> pd.DataFrame:
    """Use recorded subject selection, accepting legacy single-class inputs."""
    if raw.empty or "class_name" not in raw:
        return raw.copy()
    names = raw["class_name"].dropna().astype(str)
    names = names.loc[~names.isin(["", "nan", "None"])]
    selected = []
    if subject_class:
        selected = [subject_class]
    elif "subject_class" in raw:
        selected = [str(value) for value in raw["subject_class"].dropna().unique() if str(value)]
    if len(selected) > 1:
        raise AnalysisError("Inference CSV contains conflicting subject classes.")
    if not selected:
        if names.nunique() > 1:
            raise AnalysisError("Choose a class to analyze for this mixed-class CSV.")
        return raw.copy()
    # Keep explicit no-detection rows so downstream frame coverage remains intact.
    return raw.loc[
        raw["class_name"].eq(selected[0]) | raw["class_name"].isna() | raw["class_name"].eq("")
    ].copy()


def analysis_class_names(paths: list[str]) -> list[str]:
    """Read available classes from inference CSVs without loading masks into memory."""
    import csv

    names = set()
    for path in paths:
        if not path:
            continue
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    name = str(row.get("class_name") or "").strip()
                    if name and name not in {"nan", "None"}:
                        names.add(name)
        except (OSError, csv.Error):
            continue
    return sorted(names)
