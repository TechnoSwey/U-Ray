# High-capacity MNIST MLP (~1M parameters)

**Task:** classify all ten handwritten-digit classes in the canonical MNIST
dataset. A 1,280-unit dense hidden layer makes this a high-capacity
feed-forward image classifier and exercises U-Ray's optimized batched MLP
training path.

- Architecture: `784 -> 1,280 -> 10`
- Parameters: **1,017,610**
- Training/test split: all 60,000 train and 10,000 test examples
- Acceptance: test accuracy **>= 97%**
- Run from the repository root:
  `./uray ML/mnist_1m/train.ur --backend=native`

**Verified result:** 98.15% held-out accuracy on all 10,000 canonical MNIST
test images, after 5 epochs over the full 60,000-image training split using
the native backend and seed 42. The training script stops with an error
instead of marking the model **READY** if accuracy falls below 97%.
