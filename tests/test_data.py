import torch

from src.data import create_dataloaders


def test_mnist_dataloaders_have_expected_shapes():
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data/raw",
        batch_size=64,
    )

    images, labels = next(iter(train_loader))

    assert images.shape == (64, 1, 28, 28)
    assert labels.shape == (64,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_mnist_split_sizes():
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data/raw",
        batch_size=64,
    )

    assert len(train_loader.dataset) == 55000
    assert len(val_loader.dataset) == 5000
    assert len(test_loader.dataset) == 10000