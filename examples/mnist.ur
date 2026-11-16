import nn

// Скачай датасет командой:
// python3 examples/download_mnist.py
data = nn.load_mnist("data/mnist", true)
train = data["train"]
test = data["test"]

print("Обучающих изображений:", train["count"])
print("Тестовых изображений:", test["count"])
print("Размер изображения:", train["rows"], "x", train["columns"])

nn.rand_seed(42)
model = nn.mlp([784, 64, 10], "relu", "softmax")
nn.summary(model)

// Для полного набора используй --backend=numpy.
history = nn.train(
    model,
    train["X"],
    train["Y"],
    5,
    0.001,
    128,
    "cce",
    "adam",
    true
)

metrics = nn.evaluate(model, test["X"], test["Y"], "cce")
print("Тестовая ошибка:", metrics[0])
print("Точность на тесте:", metrics[1])

image = nn.sample(test["X"], 0)
target = nn.sample(test["Y"], 0)
scores = nn.predict(model, [image])[0]

predicted = 0
actual = 0
for i in range(1, 10) {
    if scores[i] > scores[predicted] {
        predicted = i
    }
    if target[i] > target[actual] {
        actual = i
    }
}

print("Первая тестовая цифра:", actual)
print("Предсказание U-Ray:", predicted)
print("Вероятности:", scores)

nn.save(model, "mnist_model.json")
print("Модель сохранена в mnist_model.json")
