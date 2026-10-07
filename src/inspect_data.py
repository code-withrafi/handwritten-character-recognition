from src.data import create_dataloaders


def main():
    train_loader, validation_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    print("Dataset information")
    print("-------------------")
    print(f"Training samples:   {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Test samples:       {len(test_loader.dataset)}")
    print()
    print(f"Batch image shape:  {images.shape}")
    print(f"Batch label shape:  {labels.shape}")
    print(f"Image data type:    {images.dtype}")
    print(f"Label data type:    {labels.dtype}")
    print()
    print(f"First 10 labels:    {labels[:10].tolist()}")


if __name__ == "__main__":
    main()