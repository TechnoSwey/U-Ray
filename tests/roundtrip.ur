import nn

nn.rand_seed(7)
model = nn.mlp([2, 4, 1], "leaky_relu")
path = nn.save(model, "roundtrip_model.json")
loaded = nn.load("roundtrip_model.json")
print("roundtrip equal:", model == loaded)
print("W equal:", model["layers"][0]["W"] == loaded["layers"][0]["W"])
print("act:", loaded["act"])
print("out_act:", loaded["out_act"])
