d = {"name": "U-Ray", "version": 1, "stable": true}
print(d)
print(d["name"])
print(d["version"])
print(len(d))

d["version"] = 2
d["new_key"] = [1, 2, 3]
print(d)

print(keys(d))
print(has(d, "name"))
print(has(d, "nope"))

del d["stable"]
print(d)

layer = {"W": [[1, 2], [3, 4]], "b": [0, 0]}
print(layer["W"])
print(layer["b"])

model = {"layer1": {"W": [1, 2], "b": [0]}, "layer2": {"W": [3, 4], "b": [0]}}
print(model["layer1"]["W"])
print(model["layer2"])

for k in keys(model) {
    print("слой:", k)
}
