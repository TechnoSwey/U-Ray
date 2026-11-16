import nn

// Batch normalization + dropout + residual branch on a small classifier.
nn.rand_seed(23)
X = [
    [1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8],
    [- 1.0, 0.0], [0.0, - 1.0], [- 0.8, - 0.2], [- 0.2, - 0.8]
]
Y = [
    [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
    [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]
]

model = nn.sequential([
        nn.batch_norm_layer(2),
        nn.dropout_layer(0.2),
        nn.residual_layer([
                nn.linear_layer(2, 8),
                nn.activation_layer("relu"),
                nn.linear_layer(8, 2)
        ])
])

history = nn.train_layers(model, X, Y, 20, 0.01, 4, "cce", "adam", true)
metrics = nn.evaluate_layers(model, X, Y, "cce")
predictions = nn.predict_layers(model, X, 2)

print("Final loss:", history[len(history) - 1])
print("Accuracy:", metrics[1])
print("First prediction:", predictions[0])
print("Saved model:", nn.save_layers(model, "residual_regularization.json"))
