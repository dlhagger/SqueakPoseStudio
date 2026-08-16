import ast
import subprocess
import sys
import unittest
from pathlib import Path


class PackageImportTests(unittest.TestCase):
    def test_root_core_can_be_imported_before_package_modules(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import squeakpose_core"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_lazy_project_paths_convenience_export_remains_supported(self):
        from squeakpose import ProjectPaths
        from squeakpose.project.paths import ProjectPaths as CanonicalProjectPaths

        self.assertIs(ProjectPaths, CanonicalProjectPaths)

    def test_annotation_state_import_does_not_initialize_pyqt(self):
        script = """
import builtins
import sys

real_import = builtins.__import__

def reject_pyqt(name, *args, **kwargs):
    if name == "PyQt6" or name.startswith("PyQt6."):
        raise AssertionError(f"unexpected Qt import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = reject_pyqt
from squeakpose.annotation.segmentation import SegmentationEditState
from squeakpose.annotation import PoseEditState
assert SegmentationEditState is not None
assert PoseEditState is not None
assert not any(name == "PyQt6" or name.startswith("PyQt6.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_all_project_convenience_exports_resolve(self):
        import squeakpose.project as project

        unresolved = [name for name in project.__all__ if getattr(project, name, None) is None]
        self.assertEqual(unresolved, [])

    def test_root_qt_free_compatibility_modules_preserve_function_identity(self):
        import dataset_ops
        import depth_ops
        import squeakpose_core
        from squeakpose import core
        from squeakpose import depth_ops as package_depth_ops
        from squeakpose.services import dataset_ops as package_dataset_ops

        self.assertIs(squeakpose_core.atomic_write_text, core.atomic_write_text)
        self.assertIs(depth_ops.sample_depth_map, package_depth_ops.sample_depth_map)
        self.assertIs(dataset_ops.dataset_export_paths, package_dataset_ops.dataset_export_paths)

    def test_qt_free_packages_do_not_import_repository_root_implementations(self):
        package_root = Path(__file__).resolve().parents[1] / "squeakpose"
        forbidden = {
            "analysis_ops",
            "dataset_ops",
            "depth_ops",
            "inference_ops",
            "label_io",
            "layer_ops",
            "prediction_ops",
            "segmentation_analysis_ops",
            "squeakpose_core",
        }
        violations: list[str] = []
        for relative_root in ("annotation", "project", "services", "workers"):
            for source_path in (package_root / relative_root).rglob("*.py"):
                tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported = {alias.name.split(".", 1)[0] for alias in node.names}
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported = {node.module.split(".", 1)[0]}
                    else:
                        continue
                    bad = imported & forbidden
                    if bad:
                        violations.append(f"{source_path.relative_to(package_root)}: {sorted(bad)}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
