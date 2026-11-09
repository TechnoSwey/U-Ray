import nn

nn.rand_seed(1)
model = nn.mlp([2, 1], "sigmoid")
path = nn.save(model, "v_model.json")
loaded = nn.load("v_model.json")
print("load ok:", loaded["act"] == "sigmoid")
