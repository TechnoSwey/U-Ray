# Примеры U-Ray

Здесь собраны запускаемые программы, которые показывают работу языка и ML-модуля.

Запускай команды из корня репозитория после [установки](../README.md#быстрый-старт).

| Файл | Что показывает | Запуск |
| --- | --- | --- |
| [`save_load.ur`](save_load.ur) | Обучает XOR, сохраняет модель в JSON и загружает её для предсказаний | `./uray examples/save_load.ur` |
| [`train_batch_example.ur`](train_batch_example.ur) | Создаёт синтетические данные, обучает MLP батчами с Adam и выводит метрики | `./uray examples/train_batch_example.ur` |
| [`mnist.ur`](mnist.ur) | Загружает исходные IDX-архивы, обучает MLP-классификатор цифр без NumPy, оценивает его и сохраняет модель | `python3 examples/download_mnist.py && ./uray examples/mnist.ur --backend=native` |
| [`mnist_cnn.ur`](mnist_cnn.ur) | Обучает сверточную MNIST-модель на компонуемых слоях и autodiff | `python3 examples/download_mnist.py && ./uray examples/mnist_cnn.ur --backend=numpy` |
| [`tabular_classification.ur`](tabular_classification.ur) | Читает CSV, делит данные на train/test, нормализует признаки и обучает классификатор | `./uray examples/tabular_classification.ur` |
| [`residual_regularization.ur`](residual_regularization.ur) | Обучает классификатор с BatchNorm, Dropout и residual-ветвью | `./uray examples/residual_regularization.ur` |
| [`transformer_sequence.ur`](transformer_sequence.ur) | Обучает последовательностную модель с Transformer-блоком и causal attention | `./uray examples/transformer_sequence.ur` |
| [`progress.ur`](progress.ur) | Показывает прогресс обучения и текущий loss по эпохам | `./uray examples/progress.ur` |
| [`match_struct.ur`](match_struct.ur) | Деструктурирует вложенные структуры в `match` | `./uray examples/match_struct.ur` |
| [`numpy_backend.ur`](numpy_backend.ur) | Считает матричное произведение через NumPy backend | `./uray examples/numpy_backend.ur --backend=numpy` |

В корневом [README](../README.md) есть быстрый старт, обзор возможностей и пример кода U-Ray.
