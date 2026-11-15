import nn

model = nn.mlp([1, 1], "sigmoid")
model["layers"][0]["W"][0][0] = 0.0
model["layers"][0]["b"][0] = 0.0
history = nn.train(
    model,
    [[0.0], [0.0], [0.0]],
    [[0.5], [0.5], [0.5]],
    1,
    0.0,
    2,
    "bce",
    "adam"
)
print(history[0])
