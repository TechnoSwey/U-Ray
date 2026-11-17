# Logistic perceptron

**Task:** classify two well-separated groups in 2D. The decision boundary is
linear, so a single sigmoid neuron is the appropriate model.

- Parameters: 2 weights + 1 bias = **3**
- Split: 1,000 generated training examples, 400 separately generated test
  examples
- Acceptance: test accuracy **>= 97%**
- Run from the repository root: `./uray ML/perceptron/train.ur --backend=native`

The training script uses only U-Ray and a deterministic synthetic dataset.
It trains on 1,000 examples and evaluates on 400 separately generated
examples with a different seed.

**Verified result:** 100% held-out accuracy (400/400), native backend,
100 training epochs. The script prints `RESULT: READY` only when accuracy
meets the 97% threshold and saves the checkpoint as an ignored JSON artifact.
