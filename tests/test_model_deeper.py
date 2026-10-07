import torch

from src.model_deeper import DeeperCNN


def test_deeper_cnn_output_shape():
    model = DeeperCNN()

    images = torch.randn(8, 1, 28, 28)
    outputs = model(images)

    assert outputs.shape == (8, 10)


def test_deeper_cnn_has_trainable_parameters():
    model = DeeperCNN()

    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert all(parameter.requires_grad for parameter in parameters)