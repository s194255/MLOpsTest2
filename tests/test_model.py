import pytest
import torch

from src.models.model import Classifier



def test_model():
    batch_size = 4

    X = torch.randn((batch_size, 1, 28, 28))
    model = Classifier()
    Y = model(X)

    assert Y.shape == torch.Size(
        [batch_size, 10]
    ), "model's output did not have correct shape"


def test_error_on_wrong_channel_dim():
    model = Classifier()
    x = torch.randn((4, 2, 28, 28), dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected 1 channel dim as input"):
        model(x)


def test_error_on_wrong_spatial_dim():
    model = Classifier()
    x = torch.randn((4, 1, 32, 28), dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected spatial dimension to be 28x28"):
        model(x)


def test_error_on_wrong_ndim():
    model = Classifier()
    x = torch.randn((1, 4, 1, 28, 28), dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected 4D tensor"):
        model(x)


def test_error_on_wrong_dtype():
    model = Classifier()
    x = torch.randn((4, 1, 28, 28), dtype=torch.float64)
    with pytest.raises(
        ValueError, match="Excepts dtype torch.float32 but got {}".format(x.dtype)
    ):
        model(x)
