import nn

try {
    zeros([-1])
} catch err {
    print(err)
}

t = zeros([2, 2])
try {
    print(t[2, 0])
} catch err {
    print(err)
}

try {
    t[0, -1] = 5.0
} catch err {
    print(err)
}
