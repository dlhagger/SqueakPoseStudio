import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from squeakpose.services.model_download import download_sam_weights, sam_download_error_message
from squeakpose.services.sam_models import SAM_MODELS, get_sam_model
from squeakpose.workers.sam3_download import main


class ModelDownloadTests(unittest.TestCase):
    def test_official_definitions_and_download_arguments(self):
        definitions = (
            ("sam3", "SAM 3", "facebook/sam3", "sam3.pt"),
            ("sam3.1", "SAM 3.1", "facebook/sam3.1", "sam3.1_multiplex.pt"),
        )
        for key, name, repo, filename in definitions:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                model = get_sam_model(key)
                self.assertEqual(
                    (model.display_name, model.repo_id, model.filename), (name, repo, filename)
                )
                self.assertEqual(model.access_url, f"https://huggingface.co/{repo}")
                calls = []

                def downloader(**kwargs):
                    calls.append(kwargs)
                    target = Path(kwargs["local_dir"], kwargs["filename"])
                    target.write_bytes(b"weights")
                    return str(target)

                result = download_sam_weights(root, key, downloader=downloader)
                self.assertEqual(result, os.path.join(root, filename))
                self.assertEqual(
                    calls, [{"repo_id": repo, "filename": filename, "local_dir": root}]
                )

    def test_rejects_missing_wrong_empty_and_escaped_output(self):
        for key, model in SAM_MODELS.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                target = Path(root, model.filename)
                with self.assertRaises(RuntimeError):
                    download_sam_weights(root, key, downloader=lambda **_: str(target))
                target.touch()
                for output in (
                    None,
                    "",
                    str(Path(root, "wrong.pt")),
                    str(Path(root).parent / model.filename),
                ):
                    with self.subTest(output=output), self.assertRaises(RuntimeError):
                        download_sam_weights(root, key, downloader=lambda **_: output)
                target.unlink()
                outside = Path(root, "outside.pt")
                outside.touch()
                target.symlink_to(outside)
                with self.assertRaises(RuntimeError):
                    download_sam_weights(root, key, downloader=lambda **_: str(target))

    def test_invalid_models_are_rejected_before_download(self):
        for key in ("sam2", "facebook/sam3", "../sam3.pt", ""):
            with self.subTest(key=key), self.assertRaises(ValueError):
                download_sam_weights("/tmp", key, downloader=lambda **_: self.fail("called"))

    def test_destination_must_exist(self):
        with self.assertRaises(ValueError):
            download_sam_weights("", "sam3.1")

    def test_gated_error_identifies_model_access_and_authentication(self):
        for key, model in SAM_MODELS.items():
            for error in ("403 Forbidden: gated repo", "401 Unauthorized"):
                message = sam_download_error_message(error, key)
                self.assertIn(model.display_name, message)
                self.assertIn(model.access_url, message)
                self.assertIn("hf auth login", message)
                self.assertIn("HF_TOKEN", message)

    def test_worker_rejects_unsupported_variant_and_arbitrary_repo(self):
        for extra in (["--model", "sam2"], ["--model", "sam3", "--repo-id", "other/repo"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["--destination", "/tmp", *extra])
            self.assertEqual(raised.exception.code, 2)

    def test_worker_returns_path_for_each_variant(self):
        for key, model in SAM_MODELS.items():
            target = f"/tmp/{model.filename}"
            with (
                patch(
                    "squeakpose.workers.sam3_download.download_sam_weights", return_value=target
                ) as download,
                patch("squeakpose.workers.sam3_download.write_event") as event,
            ):
                self.assertEqual(main(["--destination", "/tmp", "--model", key]), 0)
                download.assert_called_once_with("/tmp", key)
                event.assert_called_once_with({"event": "result", "model_path": target})


if __name__ == "__main__":
    unittest.main()
