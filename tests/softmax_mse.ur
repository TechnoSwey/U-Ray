import nn

model = nn.mlp([1, 2], "sigmoid", "softmax")
model["layers"][0]["W"][0][0] = 0.0
model["layers"][0]["W"][1][0] = 0.0
model["layers"][0]["b"][0] = 0.0
model["layers"][0]["b"][1] = 0.0

cache = nn.forward_mlp(model, [1.0])
grad = nn.mse_grad(cache[len(cache) - 1], [1.0, 0.0])
grads = nn.backward_mlp(model, cache, grad)
print("grad_W:", grads[0][0])
print("grad_b:", grads[0][1])
