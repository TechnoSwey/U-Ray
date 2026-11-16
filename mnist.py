"""Compact reader for the original gzip-compressed MNIST IDX files."""

import gzip
import struct
import zlib
from pathlib import Path


class _MNISTStorage:
    __slots__ = ("images", "labels", "count", "rows", "columns")

    def __init__(self, images, labels, count, rows, columns):
        self.images = images
        self.labels = labels
        self.count = count
        self.rows = rows
        self.columns = columns


class MNISTView:
    """Lazy image or one-hot-label view over compact byte storage."""

    __slots__ = ("storage", "kind")

    def __init__(self, storage, kind):
        if kind not in ("images", "labels"):
            raise ValueError("MNIST view kind must be 'images' or 'labels'")
        self.storage = storage
        self.kind = kind

    def __len__(self):
        return self.storage.count

    def sample(self, index):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("MNIST sample index must be an integer")
        if index < 0 or index >= self.storage.count:
            raise IndexError("MNIST sample index is out of range")

        if self.kind == "labels":
            target = [0.0] * 10
            target[self.storage.labels[index]] = 1.0
            return target

        image_size = self.storage.rows * self.storage.columns
        start = index * image_size
        return [
            pixel / 255.0
            for pixel in self.storage.images[start:start + image_size]
        ]

    def batch(self, indices, numpy):
        """Return a normalized float batch or a one-hot target batch."""
        index_array = numpy.asarray(indices, dtype=numpy.intp)
        if index_array.ndim != 1:
            raise ValueError("MNIST batch indices must be one-dimensional")
        if index_array.size and (
            numpy.any(index_array < 0)
            or numpy.any(index_array >= self.storage.count)
        ):
            raise IndexError("MNIST batch index is out of range")

        if self.kind == "labels":
            labels = numpy.frombuffer(
                self.storage.labels, dtype=numpy.uint8
            )[index_array]
            targets = numpy.zeros((len(index_array), 10), dtype=numpy.float32)
            targets[numpy.arange(len(index_array)), labels] = 1.0
            return targets

        image_size = self.storage.rows * self.storage.columns
        images = numpy.frombuffer(
            self.storage.images, dtype=numpy.uint8
        ).reshape(self.storage.count, image_size)
        return images[index_array].astype(numpy.float32) / 255.0


def _read_idx_file(directory, stem):
    compressed = directory / f"{stem}.gz"
    uncompressed = directory / stem
    path = compressed if compressed.is_file() else uncompressed
    if not path.is_file():
        raise FileNotFoundError(
            f"не найден {compressed.name} или {uncompressed.name} "
            f"в каталоге {directory}"
        )

    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as source:
            return source.read()
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(f"не удалось прочитать {path.name}: {error}") from error


def _load_split(directory, prefix):
    image_bytes = _read_idx_file(
        directory, f"{prefix}-images-idx3-ubyte"
    )
    label_bytes = _read_idx_file(
        directory, f"{prefix}-labels-idx1-ubyte"
    )

    if len(image_bytes) < 16:
        raise ValueError(f"{prefix}: заголовок IDX изображений неполный")
    image_magic, image_count, rows, columns = struct.unpack(
        ">IIII", image_bytes[:16]
    )
    if image_magic != 2051:
        raise ValueError(
            f"{prefix}: неверная сигнатура IDX изображений {image_magic}"
        )
    if image_count <= 0 or rows <= 0 or columns <= 0:
        raise ValueError(f"{prefix}: пустые или неверные размеры изображений")
    if rows != 28 or columns != 28:
        raise ValueError(
            f"{prefix}: ожидались изображения 28x28, получены {rows}x{columns}"
        )
    expected_image_bytes = image_count * rows * columns
    if len(image_bytes) - 16 != expected_image_bytes:
        raise ValueError(
            f"{prefix}: ожидалось {expected_image_bytes} байт изображений, "
            f"получено {len(image_bytes) - 16}"
        )

    if len(label_bytes) < 8:
        raise ValueError(f"{prefix}: заголовок IDX меток неполный")
    label_magic, label_count = struct.unpack(">II", label_bytes[:8])
    if label_magic != 2049:
        raise ValueError(
            f"{prefix}: неверная сигнатура IDX меток {label_magic}"
        )
    if label_count != image_count:
        raise ValueError(
            f"{prefix}: число меток {label_count} не совпадает "
            f"с числом изображений {image_count}"
        )
    labels = label_bytes[8:]
    if len(labels) != label_count:
        raise ValueError(
            f"{prefix}: ожидалось {label_count} меток, получено {len(labels)}"
        )
    if any(label > 9 for label in labels):
        raise ValueError(f"{prefix}: метка должна быть цифрой от 0 до 9")

    return _MNISTStorage(
        image_bytes[16:], labels, image_count, rows, columns
    )


def load_mnist_directory(directory):
    """Load train and test splits from standard MNIST IDX[.gz] files."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"каталог MNIST не найден: {root}")
    return {
        "train": _load_split(root, "train"),
        "test": _load_split(root, "t10k"),
    }
