import os
import tempfile
import unittest

from squeakpose.services.model_download import (
    SAM3_FILENAME,
    SAM3_REPO_ID,
    download_sam3_weights,
    sam3_download_error_message,
)


class ModelDownloadTests(unittest.TestCase):
    def test_downloads_official_sam3_checkpoint_into_destination(self):
        calls = []
        with tempfile.TemporaryDirectory() as root:

            def downloader(**kwargs):
                calls.append(kwargs)
                target = os.path.join(kwargs["local_dir"], kwargs["filename"])
                with open(target, "wb") as stream:
                    stream.write(b"weights")
                return target

            result = download_sam3_weights(root, downloader=downloader)

            self.assertEqual(result, os.path.join(root, SAM3_FILENAME))
            self.assertEqual(
                calls,
                [
                    {
                        "repo_id": SAM3_REPO_ID,
                        "filename": SAM3_FILENAME,
                        "local_dir": root,
                    }
                ],
            )

    def test_rejects_missing_output(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "without creating sam3.pt"):
                download_sam3_weights(
                    root,
                    downloader=lambda **_kwargs: os.path.join(root, SAM3_FILENAME),
                )

    def test_gated_error_explains_access_and_authentication(self):
        message = sam3_download_error_message("403 Forbidden: gated repo")
        self.assertIn("gated", message)
        self.assertIn("hf auth login", message)
        self.assertIn("https://huggingface.co/facebook/sam3", message)


if __name__ == "__main__":
    unittest.main()
