import nn

data = nn.load_mnist("../../examples/data/mnist", true)
train = data["train"]
test = data["test"]

nn.rand_seed(42)
model = nn.sequential([
    nn.conv2d_layer(1, 8, 3, 1, 1),
    nn.activation_layer("relu"),
    nn.max_pool2d_layer(2, 2),
    nn.flatten_layer(),
    nn.linear_layer(8 * 14 * 14, 10)
])

nn.train_layers(
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
accuracy = metrics[1]
print("Held-out test accuracy:", accuracy)

if accuracy < 0.97 {
    throw "MNIST CNN did not reach 97% test accuracy"
}

print("RESULT: READY")
nn.save_layers(model, "trained_model.json")
