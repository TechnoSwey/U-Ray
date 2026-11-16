import nn

nn.rand_seed(7)
data = nn.load_csv(
    "tabular_classification.csv",
    "class",
    0.2,
    7,
    true,
    true,
    true
)
train = data["train"]
test = data["test"]
model = nn.mlp([2, 8, 2], "relu", "softmax")
history = nn.train(
    model,
    train["X"],
    train["Y"],
    40,
    0.03,
    4,
    "cce",
    "adam",
    true
)
metrics = nn.evaluate(model, test["X"], test["Y"], "cce")

print("classes:", data["classes"])
print("final loss:", history[39])
print("test loss:", metrics[0])
print("test accuracy:", metrics[1])
