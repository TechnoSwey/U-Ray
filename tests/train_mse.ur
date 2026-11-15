import nn

model = nn.mlp([1, 1], "sigmoid")
model["layers"][0]["W"][0][0] = 0.0
model["layers"][0]["b"][0] = 0.0
nn.train(model, [[0.0]], [[0.0]], 1, 1.0, 1, "mse", "sgd")
print(model["layers"][0]["b"][0])
