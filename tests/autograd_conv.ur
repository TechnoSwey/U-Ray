import nn

nn.rand_seed(11)

model = nn.sequential([
    nn.conv2d_layer(1, 2, 3, 1, 1),
    nn.activation_layer("relu"),
    nn.max_pool2d_layer(2, 2),
    nn.flatten_layer(),
    nn.linear_layer(8, 3)
])

image = nn.tensor([[[[
    0.0, 1.0, 2.0, 3.0
], [
    4.0, 5.0, 6.0, 7.0
], [
    8.0, 9.0, 10.0, 11.0
], [
    12.0, 13.0, 14.0, 15.0
]]]])
logits = nn.forward_layers(model, image)
loss = nn.cross_entropy(logits, [[0.0, 1.0, 0.0]])
nn.backward(loss)
params = nn.parameters(model)
nn.sgd_update(params, 0.001)
nn.zero_grad(params)

print("CNN forward/backward/update:", nn.item(loss) > 0.0)
print("CNN logits shape:", nn.shape(nn.forward_layers(model, image)))
