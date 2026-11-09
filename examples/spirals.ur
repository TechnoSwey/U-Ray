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

print("точек:", len(X))

model = nn.mlp([2, 8, 8, 1], "leaky_relu")
lr = 0.1
epochs = 5000

for epoch in range(epochs) {
    total_loss = 0.0
    correct = 0
    for i in range(len(X)) {
        x = X[i]
        y = [Y[i]]
        cache = nn.forward_mlp(model, x)
        yhat = cache[len(cache) - 1]
        loss = nn.bce(yhat, y)
        total_loss = total_loss + loss
        if yhat[0] >= 0.5 {
            if Y[i] == 1.0 { correct = correct + 1 }
        } else {
            if Y[i] == 0.0 { correct = correct + 1 }
        }
        grads = nn.backward_mlp(model, cache, nn.bce_grad(yhat, y))
        nn.sgd_step(model, grads, lr)
    }
    if epoch % 500 == 0 {
        print("epoch", epoch, "loss", total_loss, "acc", correct / len(X))
    }
}

print("=== финал ===")
total_loss = 0.0
correct = 0
for i in range(len(X)) {
    cache = nn.forward_mlp(model, X[i])
    yhat = cache[len(cache) - 1]
    total_loss = total_loss + nn.bce(yhat, [Y[i]])
    if yhat[0] >= 0.5 {
        if Y[i] == 1.0 { correct = correct + 1 }
    } else {
        if Y[i] == 0.0 { correct = correct + 1 }
    }
}
print("loss:", total_loss)
print("accuracy:", correct / len(X))

saved = nn.save(model, "spirals_model.json")
print("сохранено в", saved)
