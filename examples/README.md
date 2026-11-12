# Примеры U-Ray

Здесь собраны запускаемые программы, которые показывают работу языка и ML-модуля.

Запускай команды из корня репозитория после [установки](../README.md#быстрый-старт).

| Файл | Что показывает | Запуск |
| --- | --- | --- |
| [`save_load.ur`](save_load.ur) | Обучает XOR, сохраняет модель в JSON и загружает её для предсказаний | `./uray examples/save_load.ur` |
| [`train_batch_example.ur`](train_batch_example.ur) | Создаёт синтетические данные, обучает MLP батчами с Adam и выводит метрики | `./uray examples/train_batch_example.ur` |
| [`progress.ur`](progress.ur) | Показывает прогресс обучения и текущий loss по эпохам | `./uray examples/progress.ur` |

В корневом [README](../README.md) есть быстрый старт, обзор возможностей и пример кода U-Ray.
