import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from errors import RuntimeErrorUray
from modules import ModuleRegistry


class PyTorchExportTests(unittest.TestCase):
    def setUp(self):
        self.nn = ModuleRegistry().get("nn")
        self.model = self.nn.members["mlp"].fn([[2, 3, 1], "relu", "sigmoid"])

    def test_export_uses_pytorch_state_dict_layout(self):
        saved = []
        fake_torch = types.ModuleType("torch")
        fake_torch.float32 = "float32"
        fake_torch.tensor = lambda data, dtype: {
            "data": data,
            "dtype": dtype,
        }
        fake_torch.save = lambda checkpoint, path: saved.append(
            (checkpoint, path)
        )

        with tempfile.TemporaryDirectory() as directory:
            with patch("modules.values.URAY_BASE_DIR", directory):
                with patch.dict(sys.modules, {"torch": fake_torch}):
                    path = self.nn.members["export_pytorch"].fn(
                        [self.model, "model.pt"]
                    )

        self.assertEqual(path, os.path.join(directory, "model.pt"))
        self.assertEqual(len(saved), 1)
        checkpoint, saved_path = saved[0]
        self.assertEqual(saved_path, path)
        self.assertEqual(checkpoint["format"], "uray.pytorch")
        self.assertEqual(checkpoint["sizes"], [2, 3, 1])
        self.assertEqual(checkpoint["act"], "relu")
        self.assertEqual(checkpoint["out_act"], "sigmoid")
        self.assertEqual(
            set(checkpoint["state_dict"]),
            {
                "layers.0.weight",
                "layers.0.bias",
                "layers.1.weight",
                "layers.1.bias",
            },
        )
        self.assertEqual(
            len(checkpoint["state_dict"]["layers.0.weight"]["data"]), 3
        )
        self.assertEqual(
            len(checkpoint["state_dict"]["layers.0.weight"]["data"][0]), 2
        )

    def test_missing_pytorch_has_actionable_error(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(RuntimeErrorUray, "требует PyTorch"):
                self.nn.members["export_pytorch"].fn([self.model, "model.pt"])


if __name__ == "__main__":
    unittest.main()
