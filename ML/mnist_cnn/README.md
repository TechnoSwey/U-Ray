# Compact MNIST CNN

**Task:** classify all ten handwritten-digit classes in the canonical MNIST
dataset.

- Architecture: `Conv2D(1,8,3) -> ReLU -> MaxPool -> Flatten -> Dense(1568,10)`
- Parameters: **15,770**
- Training/test split: all 60,000 train and 10,000 test examples
- Acceptance: test accuracy **>= 97%**
- Run from the repository root:
  `./uray ML/mnist_cnn/train.ur --backend=native`

**Verified result:** 97.72% held-out accuracy on all 10,000 canonical MNIST
test images, after 5 epochs over the full 60,000-image training split using
the native backend and seed 42. The training script stops with an error
instead of marking the model **READY** if accuracy falls below 97%.
