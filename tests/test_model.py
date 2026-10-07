import torch

from src.model import SimpleCNN


def test_simple_cnn_output_shape():
    model = SimpleCNN()

    images = torch.randn(8, 1, 28, 28)
    outputs = model(images)

    assert outputs.shape == (8, 10)


def test_simple_cnn_has_trainable_parameters():
    model = SimpleCNN()

    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert all(parameter.requires_grad for parameter in parameters)