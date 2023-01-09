# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


def create_tensors(files):
    X, Y = [], []
    for file in files:
        data = np.load(file)
        X.append(data["images"])
        Y.append(data["labels"])

    X = np.concatenate(X, axis=0)
    X = torch.tensor(X, dtype=torch.float32)
    X = standardize(X, 0)
    X = X.unsqueeze(1)

    Y = np.concatenate(Y, axis=0)
    Y = torch.tensor(Y)

    return X, Y


def standardize(tensor, dim):
    mean = torch.mean(tensor, dim=dim)
    std = torch.std(tensor, dim=dim)
    normalized = (tensor - mean) / (std + 000.1)
    return normalized


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    print("hej")
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    test_files = [
        os.path.join(input_filepath, "train_{}.npz".format(i)) for i in range(5)
    ]
    Xtrain, Ytrain = create_tensors(test_files)
    torch.save(Xtrain, os.path.join(output_filepath, "Xtrain.pt"))
    torch.save(Ytrain, os.path.join(output_filepath, "Ytrain.pt"))

    test_files = [os.path.join(input_filepath, "test.npz")]
    Xtest, Ytest = create_tensors(test_files)
    torch.save(Xtest, os.path.join(output_filepath, "Xtest.pt"))
    torch.save(Ytest, os.path.join(output_filepath, "Ytest.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
