import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import create_dataloaders
from src.model import SimpleCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    train_loader, validation_loader, test_loader = create_dataloaders(
        batch_size=64
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    print("\nStarting training...\n")

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy * 100:.2f}% | "
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy * 100:.2f}%"
        )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nFinal Test Results")
    print("------------------")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Create output directories
    Path("results").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)

    # Save training history
    history_path = Path("experiments/simple_cnn_history.json")

    with history_path.open("w") as file:
        json.dump(history, file, indent=4)

    # Save trained model
    model_path = Path("results/simple_cnn.pth")
    torch.save(model.state_dict(), model_path)

    # Save experiment summary
    summary_path = Path("experiments/simple_cnn_results.txt")

    with summary_path.open("w") as file:
        file.write("Experiment: Simple CNN Baseline\n\n")
        file.write("Architecture:\n")
        file.write("- Conv2D: 1 -> 16 channels\n")
        file.write("- ReLU\n")
        file.write("- MaxPool2D\n")
        file.write("- Conv2D: 16 -> 32 channels\n")
        file.write("- ReLU\n")
        file.write("- MaxPool2D\n")
        file.write("- Flatten\n")
        file.write("- Linear: 1568 -> 10\n\n")

        file.write("Training Configuration:\n")
        file.write("- Optimizer: Adam\n")
        file.write("- Learning rate: 0.001\n")
        file.write("- Batch size: 64\n")
        file.write("- Epochs: 5\n\n")

        file.write(f"Test Loss: {test_loss:.4f}\n")
        file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

    print("\nSaved files:")
    print(f"- {history_path}")
    print(f"- {model_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()

