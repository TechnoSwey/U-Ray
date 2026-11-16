import nn

nn.rand_seed(7)

model = nn.sequential([
    nn.linear_layer(2, 4),
    nn.activation_layer("tanh"),
    nn.linear_layer(4, 1)
])
params = nn.parameters(model)
x = nn.tensor([1.0, -1.0])
target = [1.0]

before = nn.item(nn.mse(nn.forward_layers(model, x), target))
for step in range(25) {
    prediction = nn.forward_layers(model, x)
    loss = nn.mse(prediction, target)
    nn.backward(loss)
    nn.sgd_update(params, 0.05)
    nn.zero_grad(params)
}
after = nn.item(nn.mse(nn.forward_layers(model, x), target))

print("autograd graph lowers loss:", after < before)
print("sequential output shape:", nn.shape(nn.forward_layers(model, x)))

vector = nn.parameter([2.0, -1.0])
operator_loss = nn.mean((vector ** 2 - 3.0 * vector) / 2.0)
nn.backward(operator_loss)
print("tensor operators build gradients:", nn.shape(nn.grad(vector)) == [2])
