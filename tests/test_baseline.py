import numpy as np

from src.baseline import (
    evaluate_baseline,
    prepare_features,
    train_baseline,
)


def test_prepare_features_flattens_images():
    images = np.zeros((4, 1, 28, 28), dtype=np.float32)

    features = prepare_features(images)

    assert features.shape == (4, 784)
    assert features.dtype == np.float32


def test_train_baseline_returns_trained_model():
    images = np.random.rand(20, 1, 28, 28).astype(np.float32)
    labels = np.array([0, 1] * 10)

    model = train_baseline(images, labels)

    assert hasattr(model, "predict")


def test_evaluate_baseline_returns_accuracy():
    images = np.random.rand(20, 1, 28, 28).astype(np.float32)
    labels = np.array([0, 1] * 10)

    model = train_baseline(images, labels)
    accuracy = evaluate_baseline(model, images, labels)

    assert 0.0 <= accuracy <= 1.0