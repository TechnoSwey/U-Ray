import math
import nn

nn.rand_seed(42)

N = 100
X = []
Y = []

for i in range(N) {
    a = math.rand() * 2.0 - 1.0
    b = math.rand() * 2.0 - 1.0
    push(X, [a, b])
    label = 0.0
    if a * a + b * b > 0.5 { label = 1.0 }
    push(Y, [label])
}

model = nn.mlp([2, 16, 16, 1], "leaky_relu")
epochs = 50

for epoch in range(epochs) {
    cache_history = nn.train(model, X, Y, 1, 0.01, 32, "bce", "adam")
    progress(epoch + 1, epochs, cache_history[0])
}

print("done")
