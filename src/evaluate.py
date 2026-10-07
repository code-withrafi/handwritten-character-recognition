from pathlib import Path

import torch
from torchvision.utils import save_image

from src.data import create_dataloaders
from src.model import SimpleCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(batch_size=64)

    model = SimpleCNN().to(device)

    model_path = Path("results/simple_cnn.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mistakes = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = probabilities.max(dim=1)

            for i in range(len(labels)):
                if predictions[i] != labels[i]:
                    mistakes.append(
                        {
                            "image": images[i].cpu(),
                            "true_label": labels[i].item(),
                            "predicted_label": predictions[i].item(),
                            "confidence": confidences[i].item(),
                        }
                    )

    mistakes.sort(key=lambda item: item["confidence"])

    worst_mistakes = mistakes[:20]

    output_dir = Path("results/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTotal incorrect predictions: {len(mistakes)}")
    print("Saving 20 lowest-confidence mistakes...\n")

    for index, mistake in enumerate(worst_mistakes, start=1):
        image_path = output_dir / f"mistake_{index:02d}.png"

        # Undo MNIST normalization before saving.
        image = mistake["image"] * 0.3081 + 0.1307
        image = image.clamp(0, 1)

        save_image(image, image_path)

        print(
            f"{index:02d}. "
            f"True: {mistake['true_label']} | "
            f"Predicted: {mistake['predicted_label']} | "
            f"Confidence: {mistake['confidence'] * 100:.2f}%"
        )

    print("\nSaved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
