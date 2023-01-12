import os
import random

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.data_traditional import MNIST
from tests import _PATH_DATA


def get_random_shape(dataset):
    img, label = random.choice(dataset)
    return img.shape, label.shape


def dataloader_working(dataset, batch_size):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    try:
        x, y = dataloader.__iter__().__next__()
        assert x.shape == torch.Size([batch_size, 1, 28, 28])
        assert y.shape == torch.Size([batch_size])
        assert x.dtype == torch.float32
        assert y.dtype == torch.int64
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_PATH_DATA, "raw")), reason="Data files not found"
)
def test_data():
    tasks = ["train", "test"]
    root = os.path.join(_PATH_DATA, "raw")
    Ns = {"train": 25000, "test": 5000}
    for task in tasks:
        dataset = MNIST(root, task)
        assert len(dataset) == Ns[task], "len of dataset was wrong"
        random_img_shape, random_label_shape = get_random_shape(dataset)
        assert random_img_shape == torch.Size(
            [1, 28, 28]
        ), "wrong shape of img in dataset"
        assert random_label_shape == torch.Size([]), "wrong shape of label in dataset"
        assert set(dataset.get_unique_labels().tolist()) == set(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ), "not all classes are represented"
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            assert dataloader_working(dataset, batch_size), "dataloader is not working"
