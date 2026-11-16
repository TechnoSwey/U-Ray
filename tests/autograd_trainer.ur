import nn

nn.rand_seed(4)

model = nn.sequential([
    nn.linear_layer(2, 4),
    nn.activation_layer("tanh"),
    nn.linear_layer(4, 2)
])
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]

history = nn.train_layers(model, X, Y, 12, 0.1, 4, "cce", "adam", false)
metrics = nn.evaluate_layers(model, X, Y)
predictions = nn.predict_layers(model, X, 2)

print("generic training history:", len(history) == 12)
print("evaluation metric count:", len(metrics) == 2)
print("prediction count:", len(predictions) == 2)
