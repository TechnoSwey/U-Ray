import nn

data = nn.load_mnist("../../examples/data/mnist", true)
train = data["train"]
test = data["test"]

nn.rand_seed(42)
model = nn.mlp([784, 64, 10], "relu", "softmax")
nn.summary(model)

nn.train(
    model,
    train["X"],
    train["Y"],
    10,
    0.001,
    128,
    "cce",
    "adam",
    true
)

metrics = nn.evaluate(model, test["X"], test["Y"], "cce")
accuracy = metrics[1]
print("Held-out test accuracy:", accuracy)

if accuracy < 0.97 {
    throw "MNIST MLP did not reach 97% test accuracy"
}

print("RESULT: READY")
nn.save(model, "trained_model.json")
