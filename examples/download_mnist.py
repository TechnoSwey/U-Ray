"""Download and validate the four standard MNIST IDX archives."""

import sys
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "data" / "mnist"
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def download():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = OUTPUT / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"уже есть: {name}")
            continue

        temporary = target.with_name(target.name + ".part")
        url = BASE_URL + name
        print(f"скачиваю: {name}")
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    repository_root = HERE.parent
    sys.path.insert(0, str(repository_root))
    from mnist import load_mnist_directory

    splits = load_mnist_directory(OUTPUT)
    if splits["train"].count != 60000 or splits["test"].count != 10000:
        raise RuntimeError(
            "Загружен неполный набор: ожидалось 60000 train и 10000 test "
            f"изображений, получено {splits['train'].count} и "
            f"{splits['test'].count}"
        )
    print(
        "MNIST проверен:",
        f"{splits['train'].count} train,",
        f"{splits['test'].count} test",
    )
    print(f"каталог: {OUTPUT}")


if __name__ == "__main__":
    download()
