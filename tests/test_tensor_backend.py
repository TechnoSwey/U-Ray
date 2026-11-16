import subprocess
import sys
from pathlib import Path
import unittest

from errors import RuntimeErrorUray
from tensor import Tensor, set_backend


class TensorBackendTests(unittest.TestCase):
    def tearDown(self):
        set_backend("python")

    def test_python_backend_rejects_max_on_empty_tensors(self):
        set_backend("python")
        for data in ([], [[], []]):
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    RuntimeErrorUray, "max от пустого тензора"
                ):
                    Tensor(data).max()

    def test_numpy_backend_rejects_max_on_empty_tensors(self):
        try:
            set_backend("numpy")
        except RuntimeErrorUray:
            self.skipTest("NumPy не установлен")
        for data in ([], [[], []]):
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                    RuntimeErrorUray, "max от пустого тензора"
                ):
                    Tensor(data).max()

    def test_native_backend_matmul(self):
        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))
        result = Tensor([[1.0, 2.0], [3.0, 4.0]]).matmul(
            Tensor([[5.0, 6.0], [7.0, 8.0]])
        )
        self.assertEqual(result.to_list(), [[19.0, 22.0], [43.0, 50.0]])

    def test_native_backend_matmul_vectorized_tails(self):
        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))
        left = [
            [((row * 7 + col * 3) % 19 - 9) / 7.0 for col in range(13)]
            for row in range(10)
        ]
        right = [
            [((row * 5 + col * 11) % 23 - 11) / 9.0 for col in range(19)]
            for row in range(13)
        ]
        expected = Tensor._matmul(left, right)
        result = Tensor(left).matmul(Tensor(right)).to_list()
        for row, expected_row in zip(result, expected):
            for actual, expected_value in zip(row, expected_row):
                self.assertAlmostEqual(actual, expected_value, places=10)

    def test_native_backend_does_not_import_numpy(self):
        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))
        script = (
            "import sys; "
            "from tensor import set_backend, get_numpy; "
            "set_backend('native'); "
            "assert get_numpy() is None; "
            "assert 'numpy' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
