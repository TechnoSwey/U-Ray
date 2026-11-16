import nn

nn.rand_seed(19)
model = nn.sequential([
    nn.conv2d_layer(1, 2, 3, 1, 1),
    nn.activation_layer("relu"),
    nn.max_pool2d_layer(2, 2),
    nn.flatten_layer(),
    nn.linear_layer(8, 2)
])
X = [
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
     8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    [15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0,
     7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
]
Y = [[1.0, 0.0], [0.0, 1.0]]

history = nn.train_layers(
    model, X, Y, 1, 0.001, 2, "cce", "adam", false, [1, 4, 4]
)
metrics = nn.evaluate_layers(model, X, Y, "cce", [1, 4, 4])
predictions = nn.predict_layers(model, X, 2, [1, 4, 4])

print("CNN training history:", len(history) == 1)
print("CNN evaluation:", len(metrics) == 2)
print("CNN predictions:", len(predictions) == 2)
