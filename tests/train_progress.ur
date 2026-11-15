import nn

model = nn.mlp([1, 1], "sigmoid")
model["layers"][0]["W"][0][0] = 0.0
model["layers"][0]["b"][0] = 0.0
history = nn.train(
    model,
    [[0.0]],
    [[0.5]],
    2,
    0.0,
    1,
    "bce",
    "sgd",
    true
)
print("epochs:", len(history))
