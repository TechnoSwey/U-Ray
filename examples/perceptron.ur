import math
import nn

// ==== данные: линейная классификация x1 + x2 > 1 ====
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
     [0.2, 0.9], [0.9, 0.2], [0.3, 0.4], [0.7, 0.8]]
Y = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]

layer = nn.dense(2, 1)
lr = 0.5

print("=== старт ===")
print("W =", layer["W"])
print("b =", layer["b"])

epochs = 200
for epoch in range(epochs) {
    total_loss = 0.0
    for i in range(len(X)) {
        x = X[i]
        y = Y[i]

        z = nn.forward(layer, x)
        yhat = nn.sigmoid(z[0])
        err = yhat - y
        total_loss = total_loss + err * err

        for j in range(2) {
            layer["W"][0][j] = layer["W"][0][j] - lr * err * x[j]
        }
        layer["b"][0] = layer["b"][0] - lr * err
    }
    if epoch % 20 == 0 {
        print("epoch", epoch, "loss", total_loss)
    }
}

print("=== финал ===")
print("W =", layer["W"])
print("b =", layer["b"])

print("=== предсказания ===")
for i in range(len(X)) {
    z = nn.forward(layer, X[i])
    p = nn.sigmoid(z[0])
    pred = 0.0
    if p >= 0.5 {
        pred = 1.0
    }
    print("x =", X[i], "true =", Y[i], "pred =", pred, "p =", p)
}
