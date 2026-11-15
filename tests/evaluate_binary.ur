import nn

model = nn.mlp([1, 1], "sigmoid")
model["layers"][0]["W"][0][0] = 0.0
model["layers"][0]["b"][0] = 0.0
metrics = nn.evaluate(model, [[0.0], [0.0]], [[0.0], [1.0]], "bce")
print(metrics[1])
