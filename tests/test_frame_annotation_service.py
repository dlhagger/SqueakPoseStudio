import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.annotation.documents import (
    PoseAnnotationDocument,
    SegmentationAnnotationDocument,
)
from squeakpose.services.annotation_save import AnnotationSaveRequest
from squeakpose.services.frame_annotations import (
    build_pose_save_request,
    build_segmentation_save_request,
    load_pose_document,
    load_segmentation_document,
    serialize_pose_snapshot,
    serialize_segmentation_snapshot,
)


class FrameAnnotationServiceTests(unittest.TestCase):
    def test_pose_load_preserves_duplicate_malformed_and_extra_row_semantics(self):
        with TemporaryDirectory() as tmp:
            label_file = Path(tmp, "frame.txt")
            label_file.write_text(
                "not a pose row\n"
                "0 0.5 0.5 0.2 0.2 0.1 0.2 2 0.8 0.9 1\n"
                "0 0.4 0.3 0.2 0.1 0.2 0.3 1\n",
                encoding="utf-8",
            )

            result = load_pose_document(
                str(label_file),
                classes_count=1,
                canonical_names=["nose"],
                class_keypoint_lookup=[["nose"]],
                image_width=100,
                image_height=200,
            )

        self.assertIsInstance(result.document, PoseAnnotationDocument)
        self.assertEqual(result.extra_keypoint_rows, 1)
        self.assertEqual(len(result.document.typed_snapshot().annotations), 1)
        annotation = result.document.annotation(0)
        self.assertIsNotNone(annotation)
        self.assertEqual(annotation.keypoints[0].visibility, 1)
        for actual, expected in zip(annotation.box, (30.0, 50.0, 20.0, 20.0)):
            self.assertAlmostEqual(actual, expected)

    def test_load_failures_recover_to_empty_typed_documents(self):
        missing = os.path.join(os.sep, "definitely-missing", "frame.txt")

        pose = load_pose_document(
            missing,
            classes_count=1,
            canonical_names=["nose"],
            class_keypoint_lookup=[["nose"]],
            image_width=20,
            image_height=10,
        )
        segmentation = load_segmentation_document(
            missing,
            classes_count=1,
            image_width=20,
            image_height=10,
        )

        self.assertEqual(pose.document.snapshot(), {})
        self.assertEqual(pose.extra_keypoint_rows, 0)
        self.assertEqual(segmentation.snapshot(), {})

    def test_segmentation_load_keeps_last_valid_class_row_and_odd_token_recovery(self):
        with TemporaryDirectory() as tmp:
            label_file = Path(tmp, "frame.txt")
            label_file.write_text(
                "0 0.1 0.1 0.5 0.1 0.5 0.5\n"
                "1 0.1 0.1 0.5 0.1 0.5 0.5\n"
                "0 0.2 0.2 0.6 0.2 0.6 0.6 0.9\n"
                "broken\n",
                encoding="utf-8",
            )

            document = load_segmentation_document(
                str(label_file),
                classes_count=1,
                image_width=100,
                image_height=50,
            )

        self.assertIsInstance(document, SegmentationAnnotationDocument)
        annotation = document.annotation(0)
        self.assertIsNotNone(annotation)
        self.assertEqual(
            annotation.segments,
            ((20.0, 10.0), (60.0, 10.0), (60.0, 30.0)),
        )

    def test_pose_snapshot_serialization_is_detached_and_builds_transaction_input(self):
        document = PoseAnnotationDocument(
            {
                0: {
                    "bbox": {"x": 20.0, "y": 10.0, "w": 60.0, "h": 20.0},
                    "keypoints": [
                        {
                            "idx": 0,
                            "canon_idx": 0,
                            "name": "nose",
                            "x": 40.0,
                            "y": 50.0,
                            "vis": 2,
                        }
                    ],
                }
            }
        )
        snapshot = document.typed_snapshot()
        document.clear()

        text = serialize_pose_snapshot(
            snapshot,
            canonical_names=["nose"],
            image_width=200,
            image_height=100,
        )
        request = build_pose_save_request(
            snapshot,
            canonical_names=["nose"],
            image_width=200,
            image_height=100,
            project_root="/project",
            source_image_path="/source/frame.png",
            image_output_path="/project/images_all/frame.png",
            label_output_path="/project/labels_all/frame.txt",
            overlay_output_path="/project/annotations/keypoints/frame.png",
        )

        self.assertEqual(
            text,
            "0 0.250000 0.200000 0.300000 0.200000 0.200000 0.500000 2\n",
        )
        self.assertIsInstance(request, AnnotationSaveRequest)
        self.assertEqual(request.label_text, text)
        self.assertEqual(request.label_output_path, "/project/labels_all/frame.txt")

    def test_segmentation_snapshot_filters_unserializable_rows_and_builds_request(self):
        document = SegmentationAnnotationDocument(
            {
                0: {"segments": [(10.0, 5.0), (50.0, 5.0)]},
                1: {"segments": [(10.0, 5.0), (50.0, 5.0), (50.0, 25.0)]},
            }
        )
        snapshot = document.typed_snapshot()

        text = serialize_segmentation_snapshot(
            snapshot,
            image_width=100,
            image_height=50,
        )
        request = build_segmentation_save_request(
            snapshot,
            image_width=100,
            image_height=50,
            project_root="/project",
            source_image_path="/source/frame.png",
            image_output_path="/project/images_all/frame.png",
            label_output_path="/project/labels_seg_all/frame.txt",
            overlay_output_path="/project/annotations/segmentation/frame.png",
        )

        self.assertEqual(
            text,
            "1 0.100000 0.100000 0.500000 0.100000 0.500000 0.500000\n",
        )
        self.assertEqual(request.label_text, text)

    def test_service_import_does_not_load_qt(self):
        code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'PyQt6' or name.startswith('PyQt6.'):
        raise AssertionError(f'unexpected Qt import: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import squeakpose.services.frame_annotations
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd()

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
