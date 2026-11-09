import math
import nn

// ==== XOR ====
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]

// ==== слои ====
l1 = nn.dense(2, 4)   // 2 -> 4
l2 = nn.dense(4, 1)   // 4 -> 1
lr = 0.5

// ==== обучение ====
epochs = 5000
for epoch in range(epochs) {
    total_loss = 0.0
    for i in range(len(X)) {
        x = X[i]
        y = [Y[i]]

        // forward с кэшем
        cache1 = nn.forward_dense(l1, x)         // [z1, x]
        z1 = cache1[0]
        h = nn.sigmoid_vec(z1)                    // скрытая активация

        cache2 = nn.forward_dense(l2, h)         // [z2, h]
        z2 = cache2[0]
        yhat = nn.sigmoid_vec(z2)                 // выход

        // loss
        loss = nn.mse(yhat, y)
        total_loss = total_loss + loss

        // backward
        // dL/dyhat
        grad_yhat = nn.mse_grad(yhat, y)
        // dL/dz2 = grad_yhat * sigmoid'(z2)
        grad_z2 = []
        for k in range(len(z2)) {
            grad_z2_k = grad_yhat[k] * nn.sigmoid_grad(z2[k])
            push(grad_z2, grad_z2_k)
        }
        // backward по l2: [grad_W2, grad_b2, grad_h]
        bw2 = nn.backward_dense(l2, h, grad_z2)
        grad_W2 = bw2[0]
        grad_b2 = bw2[1]
        grad_h = bw2[2]

        // dL/dz1 = grad_h * sigmoid'(z1)
        grad_z1 = []
        for k in range(len(z1)) {
            grad_z1_k = grad_h[k] * nn.sigmoid_grad(z1[k])
            push(grad_z1, grad_z1_k)
        }
        // backward по l1: [grad_W1, grad_b1, grad_x] (grad_x нам не нужен)
        bw1 = nn.backward_dense(l1, x, grad_z1)
        grad_W1 = bw1[0]
        grad_b1 = bw1[1]

        // sgd-шаг
        for i2 in range(len(l1["W"])) {
            for j2 in range(len(l1["W"][i2])) {
                l1["W"][i2][j2] = l1["W"][i2][j2] - lr * grad_W1[i2][j2]
            }
            l1["b"][i2] = l1["b"][i2] - lr * grad_b1[i2]
        }
        for i2 in range(len(l2["W"])) {
            for j2 in range(len(l2["W"][i2])) {
                l2["W"][i2][j2] = l2["W"][i2][j2] - lr * grad_W2[i2][j2]
            }
            l2["b"][i2] = l2["b"][i2] - lr * grad_b2[i2]
        }
    }
    if epoch % 500 == 0 {
        print("epoch", epoch, "loss", total_loss)
    }
}

print("=== предсказания ===")
for i in range(len(X)) {
    z1 = nn.forward_dense(l1, X[i])[0]
    h = nn.sigmoid_vec(z1)
    z2 = nn.forward_dense(l2, h)[0]
    p = nn.sigmoid_vec(z2)[0]
    pred = 0.0
    if p >= 0.5 {
        pred = 1.0
    }
    print("x =", X[i], "true =", Y[i], "pred =", pred, "p =", p)
}
