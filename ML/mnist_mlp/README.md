# Compact MNIST MLP

**Task:** classify all ten handwritten-digit classes in the canonical MNIST
dataset.

- Architecture: `784 -> 64 -> 10`
- Parameters: **50,890**
- Training/test split: all 60,000 train and 10,000 test examples
- Acceptance: test accuracy **>= 97%**
- Run from the repository root:
  `./uray ML/mnist_mlp/train.ur --backend=native`

**Verified result:** 97.27% held-out accuracy on all 10,000 canonical MNIST
test images, after 10 epochs over the full 60,000-image training split using
the native backend and seed 42. The training script stops with an error
instead of marking the model **READY** if accuracy falls below 97%.
