import nn

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = nn.mlp([2, 4, 1], "sigmoid")
lr = 0.5

for epoch in range(2000) {
    for i in range(len(X)) {
        cache = nn.forward_mlp(model, X[i])
        yhat = cache[len(cache) - 1]
        grads = nn.backward_mlp(model, cache, nn.mse_grad(yhat, [Y[i]]))
        nn.sgd_step(model, grads, lr)
    }
}

print("=== обучено, сохраняем ===")
saved_path = nn.save(model, "xor_model.json")
print("сохранено в", saved_path)

loaded = nn.load("xor_model.json")
print("=== загружено, предсказания ===")
for i in range(len(X)) {
    cache = nn.forward_mlp(loaded, X[i])
    yhat = cache[len(cache) - 1]
    p = yhat[0]
    pred = 0.0
    if p >= 0.5 {
        pred = 1.0
    }
    print("x =", X[i], "true =", Y[i], "pred =", pred, "p =", p)
}
