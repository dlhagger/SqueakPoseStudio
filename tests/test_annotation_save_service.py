import os
import unittest
from tempfile import TemporaryDirectory

from squeakpose.services.annotation_save import (
    AnnotationSaveRequest,
    save_annotation_transaction,
)


class AnnotationSaveServiceTests(unittest.TestCase):
    def _request(self, root: str) -> AnnotationSaveRequest:
        return AnnotationSaveRequest(
            source_image_path=os.path.join(root, "queue", "frame.jpg"),
            image_output_path=os.path.join(root, "images", "frame.jpg"),
            label_output_path=os.path.join(root, "labels", "frame.txt"),
            overlay_output_path=os.path.join(root, "overlays", "frame.png"),
            label_text="0 0.5 0.5 0.2 0.2\n",
        )

    def test_transaction_writes_all_artifacts(self):
        with TemporaryDirectory() as tmp:
            request = self._request(tmp)
            os.makedirs(os.path.dirname(request.source_image_path))
            with open(request.source_image_path, "wb") as fh:
                fh.write(b"image")

            def render(path: str) -> bool:
                with open(path, "wb") as fh:
                    fh.write(b"overlay")
                return True

            result = save_annotation_transaction(request, render_overlay=render)

            with open(result.image_path, "rb") as fh:
                self.assertEqual(fh.read(), b"image")
            with open(result.label_path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), request.label_text)
            with open(result.overlay_path, "rb") as fh:
                self.assertEqual(fh.read(), b"overlay")

    def test_failed_commit_removes_staged_files_and_keeps_existing_targets(self):
        with TemporaryDirectory() as tmp:
            request = self._request(tmp)
            for path in (
                request.source_image_path,
                request.image_output_path,
                request.label_output_path,
                request.overlay_output_path,
            ):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(request.source_image_path, "wb") as fh:
                fh.write(b"new image")
            with open(request.image_output_path, "wb") as fh:
                fh.write(b"old image")
            with open(request.label_output_path, "w", encoding="utf-8") as fh:
                fh.write("old label\n")
            with open(request.overlay_output_path, "wb") as fh:
                fh.write(b"old overlay")

            def render(path: str) -> bool:
                with open(path, "wb") as fh:
                    fh.write(b"new overlay")
                return True

            def fail(_replacements):
                raise OSError("injected failure")

            with self.assertRaises(OSError):
                save_annotation_transaction(
                    request,
                    render_overlay=render,
                    committer=fail,
                )

            with open(request.image_output_path, "rb") as fh:
                self.assertEqual(fh.read(), b"old image")
            with open(request.label_output_path, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "old label\n")
            with open(request.overlay_output_path, "rb") as fh:
                self.assertEqual(fh.read(), b"old overlay")

            hidden = []
            for directory in ("images", "labels", "overlays"):
                hidden.extend(
                    name
                    for name in os.listdir(os.path.join(tmp, directory))
                    if name.startswith(".")
                )
            self.assertEqual(hidden, [])

    def test_empty_label_text_is_rejected_before_writes(self):
        with TemporaryDirectory() as tmp:
            request = self._request(tmp)
            invalid = AnnotationSaveRequest(
                source_image_path=request.source_image_path,
                image_output_path=request.image_output_path,
                label_output_path=request.label_output_path,
                overlay_output_path=request.overlay_output_path,
                label_text=" \n",
            )

            with self.assertRaises(ValueError):
                save_annotation_transaction(invalid, render_overlay=lambda _path: True)


if __name__ == "__main__":
    unittest.main()
