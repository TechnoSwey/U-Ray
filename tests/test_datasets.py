import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import MNIST_FILES, download_mnist_directory
from errors import RuntimeErrorUray
from modules import ModuleRegistry
from tensor import set_backend


class DatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        set_backend("python")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.nn = ModuleRegistry().get("nn")

    def tearDown(self):
        set_backend("python")
        self.temp_dir.cleanup()

    def _write_csv(self, name, rows):
        path = self.directory / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        return path

    def test_loads_named_csv_labels_and_splits_deterministically(self):
        path = self._write_csv(
            "points.csv",
            [
                ["x", "y", "class"],
                [0, 0, "left"],
                [0, 1, "left"],
                [1, 0, "left"],
                [1, 1, "right"],
                [2, 1, "right"],
                [2, 2, "right"],
                [3, 2, "right"],
                [3, 3, "right"],
                [4, 3, "right"],
                [4, 4, "right"],
            ],
        )

        args = [str(path), "class", 0.2, 13, True, True, True]
        first = self.nn.members["load_csv"].fn(args)
        second = self.nn.members["load_csv"].fn(args)

        self.assertEqual(first["train"]["count"], 8)
        self.assertEqual(first["test"]["count"], 2)
        self.assertEqual(first["features"], 2)
        self.assertEqual(first["classes"], ["left", "right"])
        self.assertEqual(first["feature_names"], ["x", "y"])
        self.assertEqual(first["train"]["X"], second["train"]["X"])
        self.assertEqual(first["test"]["Y"], second["test"]["Y"])
        self.assertTrue(all(len(row) == 2 for row in first["train"]["X"]))
        self.assertTrue(all(sum(row) == 1 for row in first["train"]["Y"]))

    def test_loads_headerless_numeric_targets(self):
        path = self._write_csv(
            "regression.csv",
            [[0, 0], [1, 2], [2, 4], [3, 6], [4, 8]],
        )
        data = self.nn.members["load_csv"].fn(
            [str(path), -1, 0.2, 3, False, False, False]
        )
        self.assertEqual(data["features"], 1)
        self.assertEqual(data["train"]["count"] + data["test"]["count"], 5)
        self.assertTrue(all(len(target) == 1 for target in data["train"]["Y"]))
        self.assertTrue(all(len(row) == 1 for row in data["test"]["X"]))

    def test_csv_dataset_can_train_and_evaluate_a_model(self):
        path = self._write_csv(
            "binary.csv",
            [
                ["x1", "x2", "label"],
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
                [2, 1, 1],
                [2, 2, 1],
                [3, 2, 1],
                [3, 3, 1],
                [4, 3, 1],
                [4, 4, 1],
            ],
        )
        data = self.nn.members["load_csv"].fn(
            [str(path), "label", 0.2, 9, True, True, True]
        )
        model = self.nn.members["mlp"].fn([[2, 4, 2], "relu", "softmax"])
        history = self.nn.members["train"].fn(
            [model, data["train"]["X"], data["train"]["Y"],
             2, 0.01, 2, "cce", "adam"]
        )
        metrics = self.nn.members["evaluate"].fn(
            [model, data["test"]["X"], data["test"]["Y"], "cce"]
        )
        self.assertEqual(len(history), 2)
        self.assertTrue(all(value >= 0 for value in history))
        self.assertGreaterEqual(metrics[1], 0)
        self.assertLessEqual(metrics[1], 1)

    def test_rejects_ragged_rows_and_invalid_labels(self):
        ragged = self._write_csv(
            "ragged.csv", [["x", "label"], [1, 0], [2, 1, 0]]
        )
        with self.assertRaisesRegex(RuntimeErrorUray, "строка 3"):
            self.nn.members["load_csv"].fn([str(ragged), "label"])

        invalid = self._write_csv(
            "invalid.csv", [["x", "label"], ["NaN", 0], [2, 1]]
        )
        with self.assertRaisesRegex(RuntimeErrorUray, "NaN"):
            self.nn.members["load_csv"].fn([str(invalid), "label"])

    def test_downloads_only_missing_mnist_archives_and_checks_splits(self):
        archive_bytes = b"fake compressed bytes"

        class Response:
            def __init__(self):
                self.payload = archive_bytes

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _size):
                payload, self.payload = self.payload, b""
                return payload

        counts = {
            "train": SimpleNamespace(count=60000),
            "test": SimpleNamespace(count=10000),
        }
        with patch("datasets.urllib.request.urlopen", side_effect=lambda *_args, **_kwargs: Response()) as urlopen:
            with patch("datasets.load_mnist_directory", return_value=counts):
                loaded = download_mnist_directory(self.directory / "mnist")
                self.assertEqual(loaded, counts)
                self.assertEqual(urlopen.call_count, len(MNIST_FILES))
                for name in MNIST_FILES:
                    self.assertEqual(
                        (self.directory / "mnist" / name).read_bytes(),
                        b"fake compressed bytes",
                    )

    def test_removes_new_archives_when_downloaded_dataset_is_invalid(self):
        class Response:
            def __init__(self):
                self.payload = b"bad archive"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _size):
                payload, self.payload = self.payload, b""
                return payload

        target = self.directory / "broken-mnist"
        with patch(
            "datasets.urllib.request.urlopen",
            side_effect=lambda *_args, **_kwargs: Response(),
        ):
            with patch(
                "datasets.load_mnist_directory",
                side_effect=ValueError("invalid IDX"),
            ):
                with self.assertRaisesRegex(ValueError, "invalid IDX"):
                    download_mnist_directory(target)
        self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
