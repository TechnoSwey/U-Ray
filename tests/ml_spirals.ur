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
    push(Y, 0.0)
    push(X, [r * math.cos(t + math.pi), r * math.sin(t + math.pi)])
    push(Y, 1.0)
}

model = nn.mlp([2, 8, 8, 1], "leaky_relu")
state = nn.adam_init(model)
lr = 0.01
for epoch in range(2000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, X[i])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.bce_grad(yhat, [Y[i]]), true)
        nn.adam_step(model, grads, state, lr)
    }
}

correct = 0
for i in range(len(X)) {
    cache = nn.forward_mlp(model, X[i])
    yhat = cache[len(cache) - 1]
    p = yhat[0]
    pred = 0.0
    if p >= 0.5 { pred = 1.0 }
    if pred == Y[i] { correct = correct + 1 }
}
acc = correct / len(X)
print("spirals accuracy > 0.9:", acc > 0.9)
