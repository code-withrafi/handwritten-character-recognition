from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def create_dataloaders(
    data_dir="data/raw",
    batch_size=64,
    validation_size=5000,
    seed=42,
):
    """
    Download MNIST and create train, validation, and test DataLoaders.

    Dataset split:
        Training:   55,000 images
        Validation:  5,000 images
        Test:       10,000 images
    """

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full_train_dataset = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = len(full_train_dataset) - validation_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader