import time
from pathlib import Path

import torch

from src.data import create_dataloaders
from src.model import SimpleCNN


def benchmark_inference(model, loader, device):
    """
    Measure model inference time without including data-loading time.
    """
    model.eval()

    total_images = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            _ = model(images)

            total_images += images.size(0)

    end_time = time.perf_counter()

    total_time = end_time - start_time

    return total_time, total_images


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders(batch_size=64)

    model = SimpleCNN().to(device)

    model_path = Path("results/simple_cnn.pth")
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    total_time, total_images = benchmark_inference(
        model,
        test_loader,
        device,
    )

    average_time = total_time / total_images
    images_per_second = total_images / total_time

    print("\nInference Benchmark")
    print("-------------------")
    print(f"Test images:          {total_images}")
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average per image:    {average_time * 1000:.4f} ms")
    print(f"Images per second:    {images_per_second:.2f}")

    Path("experiments").mkdir(exist_ok=True)

    output_path = Path("experiments/inference_results.txt")

    with output_path.open("w") as file:
        file.write("Inference Benchmark\n\n")
        file.write("Model: Simple CNN\n")
        file.write(f"Device: {device}\n")
        file.write(f"Test images: {total_images}\n")
        file.write(f"Total inference time: {total_time:.4f} seconds\n")
        file.write(f"Average time per image: {average_time * 1000:.4f} ms\n")
        file.write(f"Images per second: {images_per_second:.2f}\n")

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()