import math
import nn

nn.rand_seed(42)

N_PER_CLASS = 100
X = []
Y = []

for i in range(N_PER_CLASS) {
    t = 3.0 * math.pi * i / N_PER_CLASS
    r = 0.1 + 0.9 * i / N_PER_CLASS
    push(X, [r * math.cos(t), r * math.sin(t)])
    push(Y, [0.0])
    push(X, [r * math.cos(t + math.pi), r * math.sin(t + math.pi)])
    push(Y, [1.0])
}

model = nn.mlp([2, 16, 16, 1], "leaky_relu")
history = nn.train(model, X, Y, 500, 0.01, 32, "bce", "adam")

metrics = nn.evaluate(model, X, Y, "bce")
print("trained successfully:", metrics[0] < 0.1)
print("accuracy > 0.9:", metrics[1] > 0.9)
