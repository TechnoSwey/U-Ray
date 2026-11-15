import nn

func run_forward(layer, values) {
    return nn.forward(layer, values)
}

func run_forward_mlp(model, values) {
    return nn.forward_mlp(model, values)
}

layer = nn.dense(2, 1)
try {
    run_forward(layer, [1.0])
} catch err {
    print(err)
}

model = nn.mlp([2, 1])
try {
    run_forward_mlp(model, [1.0, 2.0, 3.0])
} catch err {
    print(err)
}
