import torch

from src.benchmark_inference import benchmark_inference
from src.model import SimpleCNN


def test_benchmark_inference_returns_positive_time_and_count():
    model = SimpleCNN()

    images = torch.randn(8, 1, 28, 28)
    loader = [(images, torch.zeros(8, dtype=torch.long))]

    total_time, total_images = benchmark_inference(
        model,
        loader,
        torch.device("cpu"),
    )

    assert total_time >= 0.0
    assert total_images == 8