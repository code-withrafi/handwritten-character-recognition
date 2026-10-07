import numpy as np
from tqdm import tqdm

from src.data import create_dataloaders
from src.baseline import train_baseline, evaluate_baseline


def loader_to_numpy(loader):
    images = []
    labels = []

    for batch_images, batch_labels in tqdm(loader, desc="Loading data"):
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())

    return np.concatenate(images), np.concatenate(labels)


def main():
    train_loader, _, test_loader = create_dataloaders()

    print("Preparing training data...")
    train_images, train_labels = loader_to_numpy(train_loader)

    print("Preparing test data...")
    test_images, test_labels = loader_to_numpy(test_loader)

    print("\nTraining Logistic Regression baseline...")
    model = train_baseline(train_images, train_labels)

    print("Evaluating baseline...")
    accuracy = evaluate_baseline(
        model,
        test_images,
        test_labels,
    )

    print(f"\nBaseline test accuracy: {accuracy:.4f}")
    print(f"Baseline test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()