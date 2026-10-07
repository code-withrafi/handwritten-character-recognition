import numpy as np
from sklearn.linear_model import LogisticRegression


def prepare_features(images):
    """
    Flatten MNIST images from (N, 1, 28, 28) to (N, 784).
    """
    return images.reshape(images.shape[0], -1).astype(np.float32)


def train_baseline(train_images, train_labels):
    """
    Train a Logistic Regression classifier on flattened MNIST images.
    """
    features = prepare_features(train_images)

    model = LogisticRegression(
        max_iter=100,
        solver="lbfgs",
        random_state=42,
    )

    model.fit(features, train_labels)

    return model


def evaluate_baseline(model, test_images, test_labels):
    """
    Evaluate the trained baseline and return accuracy.
    """
    features = prepare_features(test_images)

    accuracy = model.score(features, test_labels)

    return accuracy