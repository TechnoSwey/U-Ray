import nn

// A tiny sequence-to-sequence regression model with causal self-attention.
nn.rand_seed(53)
X = [
    [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
    [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
    [[0.5, 0.5, 0.0, 1.0], [0.0, 0.5, 0.5, 1.0]],
    [[1.0, 0.0, 0.5, 0.5], [0.5, 1.0, 0.0, 0.5]]
]
Y = [
    [[0.2, 0.4], [0.3, 0.1]],
    [[0.1, 0.6], [0.7, 0.1]],
    [[0.3, 0.2], [0.2, 0.3]],
    [[0.5, 0.2], [0.1, 0.5]]
]

model = nn.sequential([
        nn.transformer_block(4, 2, 8, 0.1, true),
        nn.linear_layer(4, 2)
])

history = nn.train_layers(model, X, Y, 10, 0.01, 2, "mse", "adam", true)
metrics = nn.evaluate_layers(model, X, Y, "mse")
predictions = nn.predict_layers(model, X, 1)

print("Final loss:", history[len(history) - 1])
print("Evaluation loss:", metrics[0])
print("First sequence prediction:", predictions[0])
