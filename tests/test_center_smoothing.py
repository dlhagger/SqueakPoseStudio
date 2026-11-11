"""Regression tests for One Euro center smoothing across detection gaps."""

import importlib.util
import math
import sys
import types
import unittest

import numpy as np
import pandas as pd

from analysis_ops import AnalysisConfig, _smooth_centers


class _LowPassFilter:
    """Mirror of ``OneEuroFilter.LowPassFilter`` from the pinned oneeurofilter==0.2.1.

    ``reset()`` deliberately keeps the last *filtered* value, exactly like the
    upstream package, so these tests exercise the behaviour the fix guards against.
    """

    def __init__(self, alpha):
        self._alpha = float(alpha)
        self._y = None
        self._s = None

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self._alpha = float(alpha)
        s = value if self._y is None else self._alpha * value + (1.0 - self._alpha) * self._s
        self._y = value
        self._s = s
        return s

    def lastFilteredValue(self):  # noqa: N802 - upstream API name
        return self._s

    def reset(self):
        self._y = None


class _OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self._freq = float(freq)
        self._mincutoff = float(mincutoff)
        self._beta = float(beta)
        self._dcutoff = float(dcutoff)
        self._x = _LowPassFilter(self._alpha(self._mincutoff))
        self._dx = _LowPassFilter(self._alpha(self._dcutoff))

    def _alpha(self, cutoff):
        te = 1.0 / self._freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        prev_x = self._x.lastFilteredValue()
        dx = 0.0 if prev_x is None else (x - prev_x) * self._freq
        edx = self._dx(dx, timestamp, alpha=self._alpha(self._dcutoff))
        cutoff = self._mincutoff + self._beta * math.fabs(edx)
        return self._x(x, timestamp, alpha=self._alpha(cutoff))

    def reset(self):
        self._x.reset()
        self._dx.reset()


def _install_filter_stub_if_missing():
    if importlib.util.find_spec("OneEuroFilter") is not None:
        return None
    module = types.ModuleType("OneEuroFilter")
    module.OneEuroFilter = _OneEuroFilter
    module.LowPassFilter = _LowPassFilter
    sys.modules["OneEuroFilter"] = module
    return module


class CenterSmoothingGapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._stub = _install_filter_stub_if_missing()

    @classmethod
    def tearDownClass(cls):
        if cls._stub is not None and sys.modules.get("OneEuroFilter") is cls._stub:
            del sys.modules["OneEuroFilter"]

    @staticmethod
    def _config():
        return AnalysisConfig(
            detections_csv="",
            output_dir="unused",
            smooth=True,
            make_plots=False,
            min_cutoff=1.0,
            beta=0.5,
            d_cutoff=1.0,
        )

    @staticmethod
    def _frame(xs, ys):
        return pd.DataFrame(
            {
                "frame_index": range(len(xs)),
                "bbox_center_x": xs,
                "bbox_center_y": ys,
            }
        )

    def test_post_gap_output_matches_fresh_segment(self):
        rng = np.random.default_rng(7)
        before = 100.0 + rng.normal(0, 2, 20)
        # Large positional jump across the gap, like a mouse re-detected elsewhere.
        after = 400.0 + rng.normal(0, 2, 20)
        gap = [math.nan] * 5
        xs = list(before) + gap + list(after)
        ys = list(before[::-1]) + gap + list(after[::-1])
        config = self._config()

        joined = _smooth_centers(self._frame(xs, ys), 30.0, config)
        isolated_before = _smooth_centers(
            self._frame(list(before), list(before[::-1])), 30.0, config
        )
        isolated_after = _smooth_centers(self._frame(list(after), list(after[::-1])), 30.0, config)

        gap_slice = slice(len(before), len(before) + len(gap))
        after_slice = slice(len(before) + len(gap), None)
        self.assertTrue(joined["bbox_center_x_euro"].iloc[gap_slice].isna().all())
        self.assertTrue(joined["bbox_center_y_euro"].iloc[gap_slice].isna().all())
        np.testing.assert_allclose(
            joined["bbox_center_x_euro"].iloc[: len(before)].to_numpy(),
            isolated_before["bbox_center_x_euro"].to_numpy(),
        )
        np.testing.assert_allclose(
            joined["bbox_center_x_euro"].iloc[after_slice].to_numpy(),
            isolated_after["bbox_center_x_euro"].to_numpy(),
        )
        np.testing.assert_allclose(
            joined["bbox_center_y_euro"].iloc[after_slice].to_numpy(),
            isolated_after["bbox_center_y_euro"].to_numpy(),
        )

    def test_post_gap_segment_is_actually_smoothed(self):
        # With a stale derivative estimate the adaptive cutoff spikes and the output
        # tracks the raw samples almost exactly; a fresh filter attenuates jitter.
        rng = np.random.default_rng(3)
        after = 400.0 + rng.normal(0, 3, 30)
        xs = [100.0, 101.0, 99.0, math.nan, math.nan] + list(after)
        config = self._config()

        smoothed = _smooth_centers(self._frame(xs, xs), 30.0, config)

        raw_tail = np.asarray(xs[5:])
        smooth_tail = smoothed["bbox_center_x_euro"].iloc[5:].to_numpy()
        self.assertLess(np.std(np.diff(smooth_tail)), np.std(np.diff(raw_tail)) * 0.75)

    def test_leading_nan_rows_do_not_change_first_segment(self):
        values = [math.nan, math.nan, 10.0, 12.0, 11.0, 13.0]
        config = self._config()
        smoothed = _smooth_centers(self._frame(values, values), 30.0, config)
        fresh = _smooth_centers(self._frame(values[2:], values[2:]), 30.0, config)
        self.assertTrue(smoothed["bbox_center_x_euro"].iloc[:2].isna().all())
        np.testing.assert_allclose(
            smoothed["bbox_center_x_euro"].iloc[2:].to_numpy(),
            fresh["bbox_center_x_euro"].to_numpy(),
        )


if __name__ == "__main__":
    unittest.main()
