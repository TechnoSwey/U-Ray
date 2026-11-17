import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from _common import assert_nested_close, run_interleaved


class BenchmarkHelpersTests(unittest.TestCase):
    def test_interleaved_runner_uses_seeded_pairs_and_excludes_warmup(self):
        calls = []

        def run(backend):
            calls.append(backend)
            return len(calls)

        result = run_interleaved(
            ("numpy", "native"),
            run,
            iterations=2,
            warmups=1,
            seed=17,
            keep_results=True,
        )

        self.assertEqual(len(calls), 6)
        self.assertEqual(len(result["numpy"]["samples"]), 2)
        self.assertEqual(len(result["native"]["samples"]), 2)
        self.assertEqual(len(result["numpy"]["results"]), 2)
        self.assertEqual(len(result["native"]["results"]), 2)

    def test_nested_numeric_outputs_allow_small_roundoff(self):
        assert_nested_close(
            [[1.0, 2.0], [3.0]],
            [[1.0 + 1e-10, 2.0], [3.0]],
        )

    def test_nested_comparison_reports_a_real_mismatch(self):
        with self.assertRaisesRegex(AssertionError, r"\$\[1\]\[0\]"):
            assert_nested_close([[1.0], [2.0]], [[1.0], [2.1]])


if __name__ == "__main__":
    unittest.main()
