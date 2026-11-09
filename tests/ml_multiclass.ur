import math
import nn

nn.rand_seed(42)

centers = [[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]
N_PER_CLASS = 50
X = []
Y = []

for c in range(3) {
    cx = centers[c][0]
    cy = centers[c][1]
    for i in range(N_PER_CLASS) {
        dx = (math.rand() - 0.5) * 1.0
        dy = (math.rand() - 0.5) * 1.0
        push(X, [cx + dx, cy + dy])
        oh = [0.0, 0.0, 0.0]
        oh[c] = 1.0
        push(Y, oh)
    }
}

model = nn.mlp([2, 16, 16, 3], "leaky_relu", "softmax")
state = nn.adam_init(model)
lr = 0.01
for epoch in range(2000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, X[i])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.cce_grad(yhat, Y[i]), true)
        nn.adam_step(model, grads, state, lr)
    }
}

correct = 0
for i in range(len(X)) {
    cache = nn.forward_mlp(model, X[i])
    yhat = cache[len(cache) - 1]
    best = 0
    for k in range(1, len(yhat)) {
        if yhat[k] > yhat[best] { best = k }
    }
    true_best = 0
    for k in range(len(Y[i])) {
        if Y[i][k] > 0.5 { true_best = k }
    }
    if best == true_best { correct = correct + 1 }
}
print("multiclass accuracy = 1.0:", correct == len(X))
