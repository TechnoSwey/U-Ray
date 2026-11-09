import math
import nn

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

model = nn.mlp([2, 4, 1], "sigmoid")
lr = 0.5
epochs = 5000

for epoch in range(epochs) {
    total_loss = 0.0
    for i in range(len(X)) {
        x = X[i]
        y = [Y[i]]
        cache = nn.forward_mlp(model, x)
        yhat = cache[len(cache) - 1]
        loss = nn.mse(yhat, y)
        total_loss = total_loss + loss
        grad_out = nn.mse_grad(yhat, y)
        grads = nn.backward_mlp(model, cache, grad_out)
        nn.sgd_step(model, grads, lr)
    }
    if epoch % 500 == 0 {
        print("epoch", epoch, "loss", total_loss)
    }
}

print("=== предсказания ===")
for i in range(len(X)) {
    cache = nn.forward_mlp(model, X[i])
    yhat = cache[len(cache) - 1]
    p = yhat[0]
    pred = 0.0
    if p >= 0.5 {
        pred = 1.0
    }
    print("x =", X[i], "true =", Y[i], "pred =", pred, "p =", p)
}
