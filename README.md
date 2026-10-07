# Handwritten Character Recognition

A machine learning project for recognizing handwritten digits from the MNIST dataset using Logistic Regression and Convolutional Neural Networks (CNNs) as part of my Machine Learning Internship at **AgenticX**.

## Overview

This project compares a traditional machine learning baseline with CNN architectures and evaluates accuracy, training behavior, classification errors, and inference speed.

### Dataset

- Dataset: MNIST
- Training: 55,000 images
- Validation: 5,000 images
- Test: 10,000 images
- Image size: 28 × 28 grayscale
- Classes: 0–9

## Models & Results

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 92.37% |
| Simple CNN | 98.46% |
| Deeper CNN | **98.94%** |

The Deeper CNN improved the Simple CNN by **0.48 percentage points**.

### Simple CNN

Architecture:

    Conv2D 1 → 16
    ReLU
    MaxPool
    Conv2D 16 → 32
    ReLU
    MaxPool
    Flatten
    Linear → 10 classes

### Deeper CNN

Architecture:

    Conv2D 1 → 16
    ReLU
    MaxPool
    Conv2D 16 → 32
    ReLU
    MaxPool
    Conv2D 32 → 64
    ReLU
    MaxPool
    Flatten
    Linear → 10 classes

## Training

The Simple CNN was trained with:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 64
- Epochs: 5
- Loss: Cross-Entropy

Final Simple CNN results:

- Training Accuracy: 99.14%
- Validation Accuracy: 98.42%
- Test Accuracy: 98.46%

Training and validation curves are generated in `results/figures/`.

## Error Analysis

The Simple CNN produced **154 incorrect predictions** out of 10,000 test images.

The 20 lowest-confidence mistakes were analyzed.

Common confusion patterns included:

- `3 → 5`
- `2 → 8`
- `4 → 9`
- `7 → 2`
- `6 → 8`

Detailed analysis is available in:

`experiments/error_analysis.md`

## Inference Benchmark

The Simple CNN was benchmarked on all 10,000 test images.

| Metric | Result |
|---|---:|
| Device | CPU |
| Total inference time | 6.6572 s |
| Average per image | 0.6657 ms |
| Throughput | 1,502.13 images/s |

Results are stored in:

`experiments/inference_results.txt`

## Project Structure

    handwritten-character-recognition/
    ├── data/
    ├── experiments/
    ├── notebooks/
    ├── results/
    ├── src/
    │   ├── data.py
    │   ├── baseline.py
    │   ├── model.py
    │   ├── model_deeper.py
    │   ├── train.py
    │   ├── train_deeper.py
    │   ├── evaluate.py
    │   └── benchmark_inference.py
    ├── tests/
    ├── README.md
    ├── requirements.txt
    └── LICENSE

## Installation

    git clone https://github.com/code-withrafi/handwritten-character-recognition.git
    cd handwritten-character-recognition

    python -m venv venv
    venv\Scripts\Activate.ps1

    pip install -r requirements.txt

## Run

Train the Simple CNN:

    python -m src.train

Train the Deeper CNN:

    python -m src.train_deeper

Generate error analysis:

    python -m src.evaluate

Run inference benchmark:

    python -m src.benchmark_inference

Run tests:

    python -m pytest -v

Current test status:

    10 passed

## Key Findings

- CNNs significantly outperform the Logistic Regression baseline.
- Increasing CNN depth improved test accuracy from **98.46% to 98.94%**.
- Remaining errors mainly involve visually similar handwritten digits.
- The best tested model achieved **98.94% test accuracy**.
- The Simple CNN achieved approximately **1,502 images/second** during CPU inference.

## License

Copyright (c) 2026 Iftekhar Ibne Masud Rafi