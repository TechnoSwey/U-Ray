# Примеры U-Ray

Коллекция программ на U-Ray, показывающих язык и ML-модуль.

## Базовые

| Файл | Что показывает |
|---|---|
| `funcs.ur` | Функции, рекурсия (факториал, Фибоначчи) |
| `arrays.ur` | Списки: литералы, индексация, push/pop, циклы |
| `dicts.ur` | Словари: ключи, вложенность, keys, has, del |
| `loops.ur` | for, while, break, continue, range |
| `strings.ur` | Строки, escape-последовательности, конкатенация |
| `modules.ur` | Модули math, linalg |

## ML

| Файл | Что показывает | Команда |
|---|---|---|
| `perceptron.ur` | Линейный классификатор (однослойный) | `./uray examples/perceptron.ur` |
| `mlp_xor.ur` | XOR через ручной backprop (2→4→1) | `./uray examples/mlp_xor.ur` |
| `mlp_api.ur` | XOR через `nn.mlp` + `nn.forward_mlp` | `./uray examples/mlp_api.ur` |
| `spirals.ur` | Две спирали, 2→8→8→1, leaky_relu | `./uray examples/spirals.ur` |
| `multiclass.ur` | 3 класса, softmax, Adam | `./uray examples/multiclass.ur` |
| `save_load.ur` | Сохранение и загрузка модели в JSON | `./uray examples/save_load.ur` |

## Что смотреть

**`perceptron.ur`** — самый простой ML-пример. Один слой, сигмоида, ручное обновление весов. Показывает, как устроен градиент.

**`mlp_xor.ur`** — многослойная сеть с **ручным** backprop. Видно, как `nn.forward_dense` / `nn.backward_dense` соединяются в цепное правило. Хорошо для понимания.

**`mlp_api.ur`** — то же, но через `nn.mlp` + `nn.forward_mlp` + `nn.backward_mlp`. В 3 раза короче, потому что backprop автоматический.

**`spirals.ur`** — нелинейная задача (две спирали), `leaky_relu`, `bce`. Доказывает, что язык тянет не только XOR.

**`multiclass.ur`** — 3 класса, `softmax` на выходе, `categorical_cross_entropy`, `Adam`. Современная архитектура.

**`save_load.ur`** — модель обучается, сохраняется в JSON, загружается обратно, предсказания совпадают. Показывает, что модели переносимы.

## Быстрый запуск

```bash
cd ~/uray
./uray examples/multiclass.ur
