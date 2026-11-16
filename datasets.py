"""Built-in dataset helpers for MNIST and numeric CSV files."""

import csv
import math
import os
import tempfile
import urllib.request
from pathlib import Path

from mnist import load_mnist_directory


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
MNIST_COUNTS = {"train": 60000, "test": 10000}
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


def _download_mnist_file(directory, name):
    target = directory / name
    temporary_path = None
    request = urllib.request.Request(
        MNIST_BASE_URL + name,
        headers={"User-Agent": "U-Ray-dataset-loader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=directory, prefix=name + ".", suffix=".part",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"{name}: превышен допустимый размер архива")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if total == 0:
            raise ValueError(f"{name}: сервер вернул пустой архив")
        temporary_path.replace(target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def download_mnist_directory(directory):
    """Download missing canonical MNIST archives and validate the complete set."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name in MNIST_FILES:
        target = root / name
        if not target.is_file() or target.stat().st_size == 0:
            _download_mnist_file(root, name)
            downloaded.append(target)

    try:
        splits = load_mnist_directory(root)
        for name, expected in MNIST_COUNTS.items():
            actual = splits[name].count
            if actual != expected:
                raise ValueError(
                    f"MNIST {name}: ожидалось {expected} примеров, "
                    f"получено {actual}"
                )
        return splits
    except Exception:
        for path in downloaded:
            path.unlink(missing_ok=True)
        raise


def load_csv_dataset(
    path,
    label_column,
    test_ratio=0.2,
    seed=42,
    normalize=False,
    one_hot=False,
    header=True,
):
    """Load a numeric-feature CSV and make a reproducible train/test split."""
    import random

    if not isinstance(test_ratio, (int, float)) or isinstance(test_ratio, bool):
        raise ValueError("test_ratio должен быть числом от 0 до 1")
    if not math.isfinite(test_ratio) or not 0 < test_ratio < 1:
        raise ValueError("test_ratio должен быть строго между 0 и 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed должен быть целым числом")
    if not isinstance(normalize, bool) or not isinstance(one_hot, bool):
        raise ValueError("normalize и one_hot должны быть true или false")
    if not isinstance(header, bool):
        raise ValueError("header должен быть true или false")

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [
                (line_number, row)
                for line_number, row in enumerate(csv.reader(stream), start=1)
                if row and any(cell.strip() for cell in row)
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"не удалось прочитать CSV {source}: {error}") from error

    if header:
        if len(rows) < 3:
            raise ValueError("CSV должен содержать заголовок и минимум две строки")
        header_line, columns = rows.pop(0)
        names = [cell.strip() for cell in columns]
        if any(not name for name in names):
            raise ValueError(f"строка {header_line}: пустое имя столбца")
        if len(set(names)) != len(names):
            raise ValueError(f"строка {header_line}: имена столбцов повторяются")
    else:
        if len(rows) < 2:
            raise ValueError("CSV должен содержать минимум две строки данных")
        names = None

    width = len(rows[0][1])
    if width < 2:
        raise ValueError("CSV должен содержать признаки и столбец метки")
    for line_number, row in rows:
        if len(row) != width:
            raise ValueError(
                f"строка {line_number}: ожидалось {width} столбцов, "
                f"получено {len(row)}"
            )

    if isinstance(label_column, str):
        if names is None:
            raise ValueError("имя столбца метки доступно только при header=true")
        if label_column not in names:
            raise ValueError(f"столбец метки {label_column!r} не найден")
        label_index = names.index(label_column)
    elif isinstance(label_column, int) and not isinstance(label_column, bool):
        label_index = label_column
        if label_index < 0:
            label_index += width
    else:
        raise ValueError("label_column должен быть именем или целым индексом")
    if label_index < 0 or label_index >= width:
        raise ValueError(
            f"индекс столбца метки вне диапазона 0..{width - 1}"
        )

    feature_indices = [index for index in range(width) if index != label_index]
    features = []
    labels = []
    for line_number, row in rows:
        feature_row = []
        for index in feature_indices:
            cell = row[index].strip()
            try:
                value = float(cell)
            except ValueError as error:
                raise ValueError(
                    f"строка {line_number}, столбец {index + 1}: "
                    f"ожидалось число, получено {cell!r}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"строка {line_number}, столбец {index + 1}: "
                    "NaN и бесконечность не поддерживаются"
                )
            feature_row.append(value)
        raw_label = row[label_index].strip()
        if not raw_label:
            raise ValueError(
                f"строка {line_number}, столбец {label_index + 1}: метка пуста"
            )
        try:
            label = float(raw_label)
        except ValueError:
            label = raw_label
        if isinstance(label, float) and not math.isfinite(label):
            raise ValueError(
                f"строка {line_number}, столбец {label_index + 1}: "
                "метка должна быть конечным числом или строкой"
            )
        features.append(feature_row)
        labels.append(label)

    indices = list(range(len(features)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, min(len(indices) - 1, int(len(indices) * test_ratio)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    if normalize:
        for column in range(len(feature_indices)):
            minimum = min(features[index][column] for index in train_indices)
            maximum = max(features[index][column] for index in train_indices)
            span = maximum - minimum
            for row in features:
                row[column] = (
                    0.0 if span == 0 else (row[column] - minimum) / span
                )

    classes = []
    if one_hot:
        for label in labels:
            if label not in classes:
                classes.append(label)
        if len(classes) < 2:
            raise ValueError("one_hot требует минимум два разных класса")
        class_indices = {label: index for index, label in enumerate(classes)}
        targets = []
        for label in labels:
            target = [0.0] * len(classes)
            target[class_indices[label]] = 1.0
            targets.append(target)
    else:
        if any(not isinstance(label, float) for label in labels):
            raise ValueError(
                "строковые метки требуют one_hot=true"
            )
        targets = [[label] for label in labels]

    def make_split(selected):
        result = {
            "X": [features[index] for index in selected],
            "Y": [targets[index] for index in selected],
            "count": len(selected),
        }
        return result

    result = {
        "train": make_split(train_indices),
        "test": make_split(test_indices),
        "features": len(feature_indices),
        "classes": classes,
    }
    if names is not None:
        result["feature_names"] = [names[index] for index in feature_indices]
    return result
