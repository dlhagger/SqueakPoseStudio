"""Multi-instance labeling contracts with unchanged YOLO training formats."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from squeakpose.annotation.documents import PoseAnnotationDocument, SegmentationAnnotationDocument
from squeakpose.services.dataset import export_dataset_transaction
from squeakpose.services.dataset_ops import dataset_export_paths
from squeakpose.services.frame_annotations import (
    load_pose_document,
    load_segmentation_document,
    serialize_pose_snapshot,
    serialize_segmentation_snapshot,
)
from squeakpose.services.prediction import plan_prediction_application
from squeakpose.services.training import infer_training_task_from_yaml


class MultiAnimalTests(unittest.TestCase):
    def test_legacy_mapping_and_selected_instance_edits_preserve_other_animals(self):
        first = {"bbox": {"x": 1, "y": 2, "w": 10, "h": 12}, "keypoints": []}
        document = PoseAnnotationDocument({0: first})
        original = document.selected_instance_id(0)
        second = document.start_instance(0)
        document[0] = {"bbox": {"x": 20, "y": 2, "w": 8, "h": 12}, "keypoints": []}
        self.assertEqual(list(document), [0])
        self.assertEqual(len(document.export_annotations()), 2)
        document.select_instance(0, original)
        document[0] = first | {"bbox": {"x": 3, "y": 2, "w": 10, "h": 12}}
        document.select_instance(0, second)
        self.assertEqual(document[0]["bbox"]["x"], 20)
        cloned = PoseAnnotationDocument(document.snapshot())
        self.assertEqual(cloned.typed_snapshot(), document.typed_snapshot())
        document.delete_annotation(0)
        self.assertEqual(len(document.all_entries()), 1)
        document.clear()
        self.assertEqual(document.all_entries(), [])

    def test_typed_snapshot_restores_both_animals_and_selection_does_not_reorder(self):
        document = SegmentationAnnotationDocument()
        document.replace_instance_entries(
            [
                {"class_id": 0, "segments": [(0, 0), (3, 0), (0, 3)]},
                {"class_id": 0, "segments": [(5, 5), (8, 5), (5, 8)]},
            ]
        )
        original = document.typed_snapshot()
        document.select_instance(0, document.instance_ids(0)[0])
        self.assertEqual(document.typed_snapshot(), original)
        document.clear()
        document.restore_typed_snapshot(original)
        self.assertEqual(document.typed_snapshot(), original)

    def test_multiple_same_class_predictions_are_opt_in(self):
        for layer, geometry in (
            ("keypoints", {"xyxy": [1, 2, 8, 9], "keypoints": [[2, 3, 0.9]]}),
            ("segmentation", {"segments": [[1, 2], [8, 2], [1, 9]]}),
        ):
            payload = {
                "layer_id": layer,
                "detections": [
                    {"class_id": 0, "confidence": 0.9, **geometry},
                    {"class_id": 0, "confidence": 0.8, **geometry},
                ],
            }
            for multiple, count in ((False, 1), (True, 2)):
                plan = plan_prediction_application(
                    payload,
                    expected_layer=layer,
                    class_names=["mouse"],
                    canonical_keypoints=["nose"],
                    multiple_instances=multiple,
                )
                self.assertEqual(len(plan.pose or plan.segmentation), count)

    def test_roundtrip_export_and_training_task_keep_both_same_class_rows(self):
        for mode, rows in (
            ("pose", "0 0.2 0.3 0.1 0.2 0.2 0.3 2\n0 0.7 0.6 0.2 0.1 0 0 0\n"),
            ("segment", "0 0.1 0.1 0.3 0.1 0.1 0.3\n0 0.6 0.6 0.8 0.6 0.6 0.8\n"),
        ):
            with self.subTest(mode=mode), TemporaryDirectory() as tmp:
                root = Path(tmp)
                images = root / "images_all"
                labels = root / "labels"
                images.mkdir()
                labels.mkdir()
                (images / "frame.png").write_bytes(b"image-copy-fixture")
                label = labels / "frame.txt"
                label.write_text(rows)
                if mode == "pose":
                    document = load_pose_document(
                        str(label),
                        classes_count=1,
                        canonical_names=["nose"],
                        class_keypoint_lookup=[["nose"]],
                        image_width=100,
                        image_height=100,
                    ).document
                    text = serialize_pose_snapshot(
                        document.typed_snapshot(),
                        canonical_names=["nose"],
                        image_width=100,
                        image_height=100,
                    )
                else:
                    document = load_segmentation_document(
                        str(label),
                        classes_count=1,
                        image_width=100,
                        image_height=100,
                    )
                    text = serialize_segmentation_snapshot(
                        document.typed_snapshot(),
                        image_width=100,
                        image_height=100,
                    )
                self.assertEqual(len(document.all_entries()), 2)
                label.write_text(text)
                result = export_dataset_transaction(
                    project_root=tmp,
                    images_all_dir=str(images),
                    labels_all_dir=str(labels),
                    final_paths=dataset_export_paths(tmp, mode),
                    train_images=["frame.png"],
                    val_images=[],
                    mode=mode,
                    classes=["mouse"],
                    keypoint_names=["nose"],
                    split_seed=42,
                )
                self.assertFalse(result.errors)
                exported = root / "datasets" / mode / "labels" / "train" / "frame.txt"
                self.assertEqual(len(exported.read_text().splitlines()), 2)
                self.assertEqual(infer_training_task_from_yaml(result.dataset_yaml_path), mode)
