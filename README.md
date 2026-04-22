# Handwritten Digits Classifier

[![CI](https://github.com/ArnavGoel03/handwritten-digits-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/ArnavGoel03/handwritten-digits-classifier/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, clean TensorFlow/Keras implementation of a CNN that classifies handwritten digits (0–9) from the MNIST dataset.

Originally written in Year 11 as part of the AI elective at Delhi Public School, R. K. Puram — this repository is the cleaned-up, reproducible version.

## What it does

Loads the 70,000-image MNIST dataset, trains a two-block convolutional neural network, and evaluates it on the held-out test set. Reaches **~99.2% test accuracy** in under a minute on CPU.

## Architecture

```
Input (28 × 28 × 1)
 → Conv2D (32, 3×3, ReLU) → MaxPool (2×2)
 → Conv2D (64, 3×3, ReLU) → MaxPool (2×2)
 → Flatten → Dense (128, ReLU) → Dropout (0.5)
 → Dense (10, Softmax)
```

Optimizer: Adam · Loss: sparse categorical cross-entropy · Batch size: 128 · Epochs: 10.

## Usage

```bash
pip install -r requirements.txt

python train.py          # trains the CNN and saves models/mnist_cnn.keras
python evaluate.py       # loads the saved model and prints test metrics + confusion matrix
python predict.py <path-to-28x28-image>   # predicts the digit in a single image
```

## Results

| Metric          | Value        |
|-----------------|--------------|
| Test accuracy   | ~99.2%       |
| Test loss       | ~0.025       |
| Params          | ~225k        |
| Training time   | ~45s on CPU  |

The confusion matrix (saved to `results/confusion_matrix.png` after `evaluate.py`) shows the usual MNIST suspects — 4↔9 and 3↔5 — as the most common confusions.

## Project structure

```
.
├── train.py          · data loading, model definition, training loop
├── evaluate.py       · metrics + confusion matrix on the test set
├── predict.py        · single-image inference CLI
├── requirements.txt
└── models/           · saved .keras checkpoints (gitignored)
```

## License

MIT.
