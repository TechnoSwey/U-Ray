import nn

s1 = nn.softmax([1000.0, 1000.0])
print("softmax large:", s1)
print("softmax sum:", s1[0] + s1[1])

s2 = nn.softmax([100000.0, 0.0])
print("softmax huge:", s2)

b1 = nn.bce([1.0], [0.0])
print("bce(1, 0):", b1 > 0.0)
b2 = nn.bce([0.0], [1.0])
print("bce(0, 1):", b2 > 0.0)
b3 = nn.bce([0.5], [0.5])
print("bce(0.5, 0.5):", b3 > 0.0)

c1 = nn.categorical_cross_entropy([1.0, 0.0], [0.0, 1.0])
print("cce edge > 0:", c1 > 0.0)

sg1 = nn.sigmoid(1000.0)
print("sigmoid(1000):", sg1)
sg2 = nn.sigmoid(-1000.0)
print("sigmoid(-1000):", sg2)

nn.rand_seed(1)
model = nn.mlp([2, 1], "sigmoid")
model["layers"][0]["W"][0][0] = 0.0
model["layers"][0]["W"][0][1] = 0.0
model["layers"][0]["b"][0] = 0.0
state = nn.adam_init(model)
zero_grads = [ [ [[0.0, 0.0]], [0.0] ] ]
nn.adam_step(model, zero_grads, state, 0.01)
w_after = model["layers"][0]["W"][0][0]
print("adam zero grad W finite:", w_after == w_after)
