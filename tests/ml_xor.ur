import nn

nn.rand_seed(7)

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = nn.mlp([2, 4, 1], "sigmoid")
state = nn.momentum_init(model)
lr = 0.5
mom = 0.9
for epoch in range(3000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, X[i])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.mse_grad(yhat, [Y[i]]))
        nn.momentum_step(model, grads, state, lr, mom)
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
print("xor accuracy = 1.0:", correct == len(X))
