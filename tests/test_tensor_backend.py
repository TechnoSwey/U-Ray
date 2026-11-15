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


if __name__ == "__main__":
    unittest.main()
