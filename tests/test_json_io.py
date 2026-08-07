import json
import os
import unittest
from tempfile import TemporaryDirectory

from squeakpose.json_io import JsonFileError, read_json_file


class JsonFileTests(unittest.TestCase):
    def test_reads_bounded_json_object(self):
        with TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"value": 7}, handle)

            self.assertEqual(
                read_json_file(path, max_bytes=1024, require_object=True),
                {"value": 7},
            )

    def test_rejects_non_object_oversized_invalid_and_non_file_inputs(self):
        with TemporaryDirectory() as tmp:
            array_path = os.path.join(tmp, "array.json")
            invalid_path = os.path.join(tmp, "invalid.json")
            with open(array_path, "w", encoding="utf-8") as handle:
                json.dump([1, 2, 3], handle)
            with open(invalid_path, "w", encoding="utf-8") as handle:
                handle.write("{invalid")

            with self.assertRaises(JsonFileError):
                read_json_file(array_path, require_object=True)
            with self.assertRaises(JsonFileError):
                read_json_file(array_path, max_bytes=2)
            with self.assertRaises(JsonFileError):
                read_json_file(invalid_path)
            with self.assertRaises(JsonFileError):
                read_json_file(tmp)

    def test_rejects_symlinked_json_file(self):
        with TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "target.json")
            link = os.path.join(tmp, "link.json")
            with open(target, "w", encoding="utf-8") as handle:
                json.dump({"value": 7}, handle)
            try:
                os.symlink(target, link)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaises(JsonFileError):
                read_json_file(link)


if __name__ == "__main__":
    unittest.main()
