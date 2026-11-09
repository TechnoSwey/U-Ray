import nn

nn.rand_seed(1)
model = nn.mlp([2, 1], "sigmoid")
model["layers"][0]["W"][0][0] = 0.0
model["layers"][0]["W"][0][1] = 0.0
model["layers"][0]["b"][0] = 0.0

cache = nn.forward_mlp(model, [1.0, 1.0])
yhat = cache[len(cache) - 1]

g1 = nn.backward_mlp(model, cache, nn.bce_grad(yhat, [1.0]), true)
print("skip true grad_W:", g1[0][0])
print("skip true grad_b:", g1[0][1])

g2 = nn.backward_mlp(model, cache, nn.bce_grad(yhat, [1.0]))
print("skip false grad_W:", g2[0][0])
print("skip false grad_b:", g2[0][1])
