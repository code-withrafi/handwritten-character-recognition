import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    history_path = Path("experiments/simple_cnn_history.json")
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    with history_path.open("r") as file:
        history = json.load(file)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    loss_path = output_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        [accuracy * 100 for accuracy in history["train_accuracy"]],
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        epochs,
        [accuracy * 100 for accuracy in history["validation_accuracy"]],
        marker="o",
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    accuracy_path = output_dir / "accuracy_curve.png"
    plt.savefig(accuracy_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Training curves saved:")
    print(f"- {loss_path}")
    print(f"- {accuracy_path}")


if __name__ == "__main__":
    main()
