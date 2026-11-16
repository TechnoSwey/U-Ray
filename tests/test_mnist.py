import gzip
import math
import struct
import tempfile
import unittest
from pathlib import Path

from errors import RuntimeErrorUray
from modules import ModuleRegistry
from tensor import set_backend


def _write_split(directory, prefix, images, labels):
    image_data = b"".join(images)
    image_header = struct.pack(">IIII", 2051, len(images), 28, 28)
    label_header = struct.pack(">II", 2049, len(labels))
    with gzip.open(
        directory / f"{prefix}-images-idx3-ubyte.gz", "wb"
    ) as output:
        output.write(image_header + image_data)
    with gzip.open(
        directory / f"{prefix}-labels-idx1-ubyte.gz", "wb"
    ) as output:
        output.write(label_header + bytes(labels))


class MNISTTests(unittest.TestCase):
    def setUp(self):
        set_backend("python")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        image_size = 28 * 28
        images = [
            bytes(image_size),
            bytes([255]) * image_size,
            bytes([128]) * image_size,
            bytes([64]) * image_size,
        ]
        _write_split(self.directory, "train", images, [0, 1, 7, 9])
        _write_split(
            self.directory,
            "t10k",
            [bytes(image_size), bytes([255]) * image_size],
            [3, 8],
        )
        self.nn = ModuleRegistry().get("nn")

    def tearDown(self):
        set_backend("python")
        self.temp_dir.cleanup()

    def _load(self):
        return self.nn.members["load_mnist"].fn([str(self.directory)])

    def _model(self):
        self.nn.members["rand_seed"].fn([17])
        return self.nn.members["mlp"].fn([[784, 5, 10], "relu", "softmax"])

    def test_loads_compressed_idx_and_normalizes_images(self):
        data = self._load()
        train = data["train"]
        self.assertEqual(train["count"], 4)
        self.assertEqual(train["rows"], 28)
        self.assertEqual(train["columns"], 28)
        self.assertEqual(train["X"].sample(0), [0.0] * 784)
        self.assertEqual(train["X"].sample(1), [1.0] * 784)
        self.assertEqual(train["X"].sample(2)[0], 128 / 255.0)
        self.assertEqual(train["Y"].sample(2), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(data["test"]["count"], 2)

    def test_loads_uncompressed_idx_files(self):
        for archive in self.directory.glob("*.gz"):
            with gzip.open(archive, "rb") as source:
                archive.with_suffix("").write_bytes(source.read())
            archive.unlink()

        data = self._load()
        self.assertEqual(data["train"]["count"], 4)
        self.assertEqual(data["test"]["count"], 2)
        self.assertEqual(data["test"]["Y"].sample(1)[8], 1.0)

    def test_training_and_evaluation_accept_lazy_mnist_views(self):
        data = self._load()
        model = self._model()
        history = self.nn.members["train"].fn([
            model,
            data["train"]["X"],
            data["train"]["Y"],
            1,
            0.01,
            2,
            "cce",
            "adam",
        ])
        metrics = self.nn.members["evaluate"].fn([
            model,
            data["test"]["X"],
            data["test"]["Y"],
            "cce",
        ])
        predictions = self.nn.members["predict"].fn([
            model, data["test"]["X"]
        ])
        first_prediction = self.nn.members["predict"].fn([
            model, data["test"]["X"], 1
        ])
        first_image = self.nn.members["sample"].fn([
            data["test"]["X"], 0
        ])

        self.assertEqual(len(history), 1)
        self.assertTrue(math.isfinite(history[0]))
        self.assertTrue(math.isfinite(metrics[0]))
        self.assertGreaterEqual(metrics[1], 0.0)
        self.assertLessEqual(metrics[1], 1.0)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(predictions[0]), 10)
        self.assertEqual(len(first_prediction), 1)
        self.assertEqual(first_image, [0.0] * 784)

    def test_rejects_corrupt_idx_header(self):
        with gzip.open(
            self.directory / "train-images-idx3-ubyte.gz", "wb"
        ) as output:
            output.write(struct.pack(">IIII", 123, 4, 28, 28))
        with self.assertRaisesRegex(RuntimeErrorUray, "сигнатура IDX"):
            self._load()

    def test_numpy_batches_match_python_training(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is optional")

        data = self._load()
        for optimizer in ("sgd", "momentum", "adam"):
            set_backend("python")
            python_model = self._model()
            self.nn.members["train"].fn([
                python_model,
                data["train"]["X"],
                data["train"]["Y"],
                1,
                0.01,
                4,
                "cce",
                optimizer,
            ])
            python_metrics = self.nn.members["evaluate"].fn([
                python_model,
                data["test"]["X"],
                data["test"]["Y"],
                "cce",
            ])

            set_backend("numpy")
            numpy_model = self._model()
            self.nn.members["train"].fn([
                numpy_model,
                data["train"]["X"],
                data["train"]["Y"],
                1,
                0.01,
                4,
                "cce",
                optimizer,
            ])
            numpy_metrics = self.nn.members["evaluate"].fn([
                numpy_model,
                data["test"]["X"],
                data["test"]["Y"],
                "cce",
            ])

            self.assertAlmostEqual(
                python_metrics[0],
                numpy_metrics[0],
                places=6,
                msg=f"CCE differs for {optimizer}",
            )
            self.assertEqual(python_metrics[1], numpy_metrics[1])

    def test_numpy_trainer_matches_other_loss_activations(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is optional")
        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))

        examples = [
            (
                [2, 4, 1], "relu", "sigmoid", "bce",
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0], [1.0], [1.0], [0.0]],
            ),
            (
                [2, 4, 1], "relu", "linear", "mse",
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0], [1.0], [1.0], [2.0]],
            ),
            (
                [2, 4, 3], "relu", "softmax", "mse",
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            ),
        ]

        for sizes, act, out_act, loss, X, Y in examples:
            metrics = []
            for backend in ("python", "numpy", "native"):
                set_backend(backend)
                self.nn.members["rand_seed"].fn([23])
                model = self.nn.members["mlp"].fn([sizes, act, out_act])
                self.nn.members["train"].fn([
                    model, X, Y, 2, 0.01, 2, loss, "adam"
                ])
                metrics.append(self.nn.members["evaluate"].fn([
                    model, X, Y, loss
                ]))
            for candidate in metrics[1:]:
                self.assertAlmostEqual(metrics[0][0], candidate[0], places=6)
                self.assertEqual(metrics[0][1], candidate[1])

    def test_native_backend_trains_and_evaluates_lazy_mnist(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is needed only as the comparison backend")
        try:
            set_backend("native")
        except RuntimeErrorUray as error:
            self.skipTest(str(error))

        data = self._load()
        results = []
        for backend in ("numpy", "native"):
            set_backend(backend)
            self.nn.members["rand_seed"].fn([17])
            model = self.nn.members["mlp"].fn([
                [784, 64, 10], "relu", "softmax"
            ])
            self.nn.members["rand_seed"].fn([29])
            history = self.nn.members["train"].fn([
                model,
                data["train"]["X"],
                data["train"]["Y"],
                2,
                0.01,
                2,
                "cce",
                "adam",
                False,
            ])
            metrics = self.nn.members["evaluate"].fn([
                model,
                data["test"]["X"],
                data["test"]["Y"],
                "cce",
            ])
            predictions = self.nn.members["predict"].fn([
                model, data["test"]["X"], 1
            ])
            results.append((history, metrics, predictions))

        for native_value, numpy_value in zip(results[1][0], results[0][0]):
            self.assertAlmostEqual(native_value, numpy_value, places=5)
        self.assertAlmostEqual(results[1][1][0], results[0][1][0], places=5)
        self.assertEqual(results[1][1][1], results[0][1][1])
        self.assertEqual(len(results[1][2]), 1)
        self.assertEqual(len(results[1][2][0]), 10)


if __name__ == "__main__":
    unittest.main()
