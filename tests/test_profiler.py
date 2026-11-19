import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from main import main
from profiler import MLProfiler


class ProfilerTests(unittest.TestCase):
    def test_report_aggregates_calls_and_marks_failures(self):
        profiler = MLProfiler()
        profiler.record("nn.matmul", 10, [[ [1.0, 2.0] ]], [[3.0]])
        profiler.record("nn.matmul", 20, [[ [1.0, 2.0] ]], [[4.0]])
        profiler.record("nn.backward", 5, [], failed=True)

        report = profiler.report("python", 100)

        self.assertEqual(report["profiled_operations"], 2)
        matmul = next(
            operation for operation in report["operations"]
            if operation["name"] == "nn.matmul"
        )
        self.assertEqual(matmul["calls"], 2)
        self.assertEqual(matmul["failures"], 0)
        self.assertAlmostEqual(matmul["total_seconds"], 30e-9)
        self.assertEqual(matmul["args"], ["list[1, 2]"])
        backward = next(
            operation for operation in report["operations"]
            if operation["name"] == "nn.backward"
        )
        self.assertEqual(backward["failures"], 1)

    def test_profile_json_includes_program_output_and_nn_calls(self):
        source = (
            "import nn\n"
            "a = nn.tensor([[1.0, 2.0]], true)\n"
            "b = nn.tensor([[3.0], [4.0]], true)\n"
            "result = nn.matmul(a, b)\n"
            "loss = nn.mean(result)\n"
            "nn.backward(loss)\n"
            "print(nn.item(loss))\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "profile.ur"
            script.write_text(source, encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main([
                    "profile", str(script), "--backend=python", "--format=json",
                ])

        self.assertEqual(status, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["program_output"], "11.0\n")
        self.assertEqual(report["backend"], "python")
        self.assertGreaterEqual(report["runtime_seconds"], 0)
        names = {operation["name"] for operation in report["operations"]}
        self.assertIn("nn.matmul", names)
        self.assertIn("nn.backward", names)

    def test_profile_cli_handles_program_without_nn_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "plain.ur"
            script.write_text('print("plain")\n', encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = main(["profile", str(script)])

        self.assertEqual(status, 0)
        self.assertIn("plain", output.getvalue())
        self.assertIn("nn.* calls: none", output.getvalue())


if __name__ == "__main__":
    unittest.main()
