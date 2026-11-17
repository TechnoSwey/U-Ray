# U-Ray ML model gallery

This directory contains complete ML programs written in U-Ray. Each model
folder documents its task, parameter count, training command, and held-out
test result.

A model is marked **READY** only when its full training script finishes and
its independent test accuracy is at least **97%**. Training and inference use
U-Ray's built-in `native` backend; NumPy is not required.

## Models

| Model | Task | Parameters | Status |
| --- | --- | ---: | --- |
| [Perceptron](perceptron/) | Linearly separable two-class data | 3 | READY — 100% test accuracy |
| [MNIST CNN](mnist_cnn/) | Handwritten digit classification | 15,770 | READY — 97.72% test accuracy |
| [MNIST MLP](mnist_mlp/) | Handwritten digit classification | 50,890 | READY — 97.27% test accuracy |
| [MNIST ~1M MLP](mnist_1m/) | High-capacity handwritten digit classifier | 1,017,610 | READY — 98.15% test accuracy |

The MNIST programs use the canonical 60,000-example training split and
10,000-example test split. Download the data once from the repository root:

```bash
python3 examples/download_mnist.py
```

Run a model from the repository root:

```bash
./uray ML/perceptron/train.ur --backend=native
```

Each training program reports the held-out accuracy and refuses the **READY**
status when it is below 97%. The verified `trained_model.json` checkpoints are
included in the repository; rerunning training updates the corresponding
tracked weights.
