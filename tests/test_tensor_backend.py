import subprocess
import sys
from pathlib import Path
import unittest

from errors import RuntimeErrorUray
from autograd import GradTensor
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

    def test_native_sum_and_mean_gradients(self):
        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))
        data = [
            [[[float(batch + channel + row + col) for col in range(3)]
              for row in range(2)]]
            for batch in range(2)
            for channel in range(2)
        ]
        tensor = GradTensor(data, requires_grad=True)
        result = tensor.mean()
        result.backward()

        expected_sum = sum(
            value
            for batch in data
            for channel in batch
            for row in channel
            for value in row
        )
        self.assertAlmostEqual(result.data, expected_sum / 24)
        self.assertEqual(tensor.grad, [[[[1.0 / 24] * 3 for _ in range(2)]]
                                      for _ in range(4)])
        self.assertIsNot(tensor.grad[0][0][0], tensor.grad[0][0][1])
        self.assertIsNot(tensor.grad[0][0][0], tensor.grad[1][0][0])
        cancellation = GradTensor([1e16, 1.0, -1e16]).sum_all()
        self.assertEqual(cancellation.data, 1.0)

    def test_native_layer_norm_matches_python_outputs_and_gradients(self):
        data = [
            [
                [float(batch * 3 + row) + feature * 0.25
                 for feature in range(4)]
                for row in range(3)
            ]
            for batch in range(2)
        ]
        gamma = [0.7, -1.2, 1.5, 0.3]
        beta = [0.1, -0.2, 0.05, 0.4]

        set_backend("python")
        expected_input = GradTensor(data, requires_grad=True)
        expected_gamma = GradTensor(gamma, requires_grad=True)
        expected_beta = GradTensor(beta, requires_grad=True)
        expected_output = expected_input.layer_norm(
            expected_gamma, expected_beta
        )
        expected_output_data = expected_output.data
        expected_output.sum_all().backward()

        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))
        actual_input = GradTensor(data, requires_grad=True)
        actual_gamma = GradTensor(gamma, requires_grad=True)
        actual_beta = GradTensor(beta, requires_grad=True)
        actual_output = actual_input.layer_norm(actual_gamma, actual_beta)
        actual_output.sum_all().backward()

        def assert_close(actual, expected):
            if isinstance(expected, list):
                self.assertIsInstance(actual, list)
                self.assertEqual(len(actual), len(expected))
                for actual_item, expected_item in zip(actual, expected):
                    assert_close(actual_item, expected_item)
            else:
                self.assertAlmostEqual(actual, expected, places=10)

        assert_close(actual_output.data, expected_output_data)
        assert_close(actual_input.grad, expected_input.grad)
        assert_close(actual_gamma.grad, expected_gamma.grad)
        assert_close(actual_beta.grad, expected_beta.grad)

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

    def test_native_matmul_rejects_non_finite_tensor_values(self):
        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))
        left = Tensor([[float("nan")]])
        right = Tensor([[1.0]])
        with self.assertRaisesRegex(RuntimeErrorUray, "конечными числами"):
            left.matmul(right)

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

    def test_native_cnn_cli_runs_without_numpy(self):
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "tests" / "autograd_cnn_trainer.ur"
        script = (
            "import sys; from main import main; "
            f"status = main([{str(script_path)!r}, '--backend=native']); "
            "assert status == 0; "
            "assert 'numpy' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CNN training history: true", result.stdout)
        self.assertIn("CNN evaluation: true", result.stdout)
        self.assertIn("CNN predictions: true", result.stdout)


if __name__ == "__main__":
    unittest.main()
