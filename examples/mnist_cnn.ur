import nn

// Сверточная классификация MNIST: Conv2D -> ReLU -> MaxPool -> Dense.
// Запускайте с --backend=native для встроенных C-ядер без NumPy.
data = nn.load_mnist("data/mnist", true)
train = data["train"]
test = data["test"]

model = nn.sequential([
        nn.conv2d_layer(1, 8, 3, 1, 1),
        nn.activation_layer("relu"),
        nn.max_pool2d_layer(2, 2),
        nn.flatten_layer(),
        nn.linear_layer(8 * 14 * 14, 10)
])

history = nn.train_layers(
    model,
    train["X"],
    train["Y"],
    5,
    0.001,
    128,
    "cce",
    "adam",
    true,
    [1, 28, 28]
)
metrics = nn.evaluate_layers(
    model, test["X"], test["Y"], "cce", [1, 28, 28]
)
predictions = nn.predict_layers(model, test["X"], 1, [1, 28, 28])

print("Test loss:", metrics[0])
print("Test accuracy:", metrics[1])
print("First prediction probabilities:", predictions[0])
print("Saved model:", nn.save_layers(model, "mnist_cnn_model.json"))
