import unittest

from squeakpose.services.tracking import resolve_tracking_config


class TrackingServiceTests(unittest.TestCase):
    def test_auto_resolves_from_expected_animal_count(self):
        self.assertEqual(resolve_tracking_config(1, "auto").resolved_tracker, "bytetrack")
        self.assertEqual(resolve_tracking_config(2, "auto").resolved_tracker, "botsort")

    def test_manual_tracker_overrides_animal_count_policy(self):
        config = resolve_tracking_config(4, "byte-track")
        self.assertEqual(config.requested_tracker, "bytetrack")
        self.assertEqual(config.resolved_tracker, "bytetrack")

    def test_count_and_tracker_are_validated(self):
        with self.assertRaises(ValueError):
            resolve_tracking_config(0)
        with self.assertRaises(ValueError):
            resolve_tracking_config(33)
        with self.assertRaises(ValueError):
            resolve_tracking_config(1, "unknown")

    def test_disabled_tracking_preserves_request_but_resolves_none(self):
        config = resolve_tracking_config(3, "botsort", enabled=False)
        self.assertFalse(config.enabled)
        self.assertEqual(config.requested_tracker, "botsort")
        self.assertEqual(config.resolved_tracker, "none")


if __name__ == "__main__":
    unittest.main()
