import math
import nn

TRAIN_PER_CLASS = 500
TEST_PER_CLASS = 200

train_x = []
train_y = []
nn.rand_seed(42)
for class_id in range(2) {
    center = -1.0
    label = 0.0
    if class_id == 1 {
        center = 1.0
        label = 1.0
    }
    for i in range(TRAIN_PER_CLASS) {
        x1 = center + (math.rand() - 0.5) * 0.5
        x2 = center + (math.rand() - 0.5) * 0.5
        push(train_x, [x1, x2])
        push(train_y, [label])
    }
}

test_x = []
test_y = []
nn.rand_seed(43)
for class_id in range(2) {
    center = -1.0
    label = 0.0
    if class_id == 1 {
        center = 1.0
        label = 1.0
    }
    for i in range(TEST_PER_CLASS) {
        x1 = center + (math.rand() - 0.5) * 0.5
        x2 = center + (math.rand() - 0.5) * 0.5
        push(test_x, [x1, x2])
        push(test_y, [label])
    }
}

nn.rand_seed(44)
model = nn.mlp([2, 1], "sigmoid", "sigmoid")
nn.summary(model)
nn.train(model, train_x, train_y, 100, 0.03, 32, "bce", "adam", true)
metrics = nn.evaluate(model, test_x, test_y, "bce")
accuracy = metrics[1]
print("Held-out test accuracy:", accuracy)

if accuracy < 0.97 {
    throw "Perceptron did not reach 97% test accuracy"
}

print("RESULT: READY")
nn.save(model, "trained_model.json")
