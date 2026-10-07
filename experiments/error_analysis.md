# Error Analysis

## Overview

The trained Simple CNN was evaluated on the MNIST test set containing 10,000 images.

The model achieved:

- **Test Accuracy:** 98.46%
- **Incorrect Predictions:** 154
- **Correct Predictions:** 9,846

To investigate the model's weaknesses, the 20 lowest-confidence incorrect predictions were selected.

These examples represent cases where the model was uncertain about its prediction even though the prediction was incorrect.

---

## 20 Lowest-Confidence Mistakes

| # | True Label | Predicted Label | Confidence |
|---|---:|---:|---:|
| 1 | 3 | 5 | 34.80% |
| 2 | 9 | 4 | 34.91% |
| 3 | 3 | 1 | 37.38% |
| 4 | 6 | 2 | 38.18% |
| 5 | 6 | 8 | 39.88% |
| 6 | 3 | 5 | 40.58% |
| 7 | 2 | 7 | 41.45% |
| 8 | 2 | 8 | 43.83% |
| 9 | 5 | 7 | 45.33% |
| 10 | 7 | 3 | 48.76% |
| 11 | 7 | 2 | 49.02% |
| 12 | 7 | 4 | 49.23% |
| 13 | 2 | 8 | 49.43% |
| 14 | 7 | 9 | 51.19% |
| 15 | 4 | 9 | 51.25% |
| 16 | 1 | 8 | 52.33% |
| 17 | 3 | 2 | 52.54% |
| 18 | 4 | 9 | 52.70% |
| 19 | 2 | 1 | 52.82% |
| 20 | 0 | 8 | 52.89% |

---

## Observations

### 1. Digit 3 is difficult in several cases

The model incorrectly classified:

- 3 → 5
- 3 → 1
- 3 → 2

The 3 → 5 confusion occurred twice among the 20 lowest-confidence mistakes.

This suggests that variations in the shape and curvature of handwritten 3s can make them resemble other digits.

### 2. Digit 7 shows several different confusions

The model incorrectly classified 7 as:

- 3
- 2
- 4
- 9

This indicates that handwritten 7s with different writing styles can be difficult to distinguish from other digits.

### 3. Digit 2 is frequently confused

The model incorrectly classified 2 as:

- 7
- 8
- 1

The 2 → 8 confusion occurred twice.

This is likely related to variations in the curves and loops used when writing the digit.

### 4. Digit 4 is confused with 9

Two of the 20 lowest-confidence mistakes were:

- 4 → 9
- 4 → 9

This suggests that some handwritten 4s have visual characteristics that overlap with 9.

### 5. Digit 6 is confused with 2 and 8

The model produced:

- 6 → 2
- 6 → 8

These digits can share similar curved structures depending on handwriting style.

---

## Confidence Analysis

The confidence values of the 20 mistakes ranged from:

- **Lowest:** 34.80%
- **Highest:** 52.89%

The lowest-confidence mistake was:

> True digit: **3**  
> Predicted digit: **5**  
> Confidence: **34.80%**

This shows that the model was relatively uncertain about this prediction.

The highest-confidence prediction among these 20 incorrect examples was:

> True digit: **0**  
> Predicted digit: **8**  
> Confidence: **52.89%**

Even though this was an incorrect prediction, the model assigned slightly more than 50% probability to its predicted class.

---

## Main Error Patterns

The main confusion patterns observed in the 20 lowest-confidence mistakes are:

- **3 → 5**
- **2 → 8**
- **4 → 9**
- **7 → 2**
- **6 → 8**
- **3 → 1**
- **7 → 4**
- **7 → 9**

These errors are understandable because handwritten digits can have substantial variation in stroke shape, curvature, spacing, and writing style.

---

## Interpretation

The error analysis indicates that the CNN performs very well overall, but its remaining errors are mainly associated with visually ambiguous handwritten digits.

The model does not appear to fail randomly. Instead, many errors occur between digits with similar visual structures.

This supports the usefulness of convolutional layers because the CNN can learn spatial patterns such as edges, curves, and local shapes. However, some unusual or ambiguous handwriting remains difficult to classify correctly.

---

## Saved Error Examples

The 20 analyzed images were automatically saved by the evaluation script.

Location:

```text
results/predictions/