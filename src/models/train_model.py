import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_traditional import MNIST
from src.models.model import Classifier, Inference


class Main:
    def __init__(
        self,
        models,
        dataloaders,
        optimizer,
        scheduler,
        criterion,
        save_path,
        weight_path=None,
    ):
        self.models = models
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.save_path = save_path

        if weight_path:
            checkpoint = torch.load(weight_path)
            self.models["train"].load_state_dict(checkpoint["model"])

        self.task_functions = {"train": self.train_fun, "test": self.test_fun}

    def train_fun(self, model, dataloader):
        epoch_losses = []
        for images, labels in dataloader:
            self.optimizer.zero_grad()
            pred = model(images)
            loss = self.criterion(pred, labels)
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.item())
        return epoch_losses

    def test_fun(self, model, dataloader):
        model.eval()
        epoch_accs = []
        for images, labels in dataloader:
            pred = model(images)
            batch_acc = torch.mean((pred == labels).to(torch.float32))
            epoch_accs.append(batch_acc.item())
        return epoch_accs

    def train(self, n_epochs):
        tasks = ["train", "test"]
        all_scores = {task: [] for task in tasks}
        for i in tqdm(range(n_epochs)):
            for task in tasks:
                scores = self.task_functions[task](
                    self.models[task], self.dataloaders[task]
                )
                all_scores[task].append(scores)
                if task == "train":
                    self.scheduler.step()
        self.plot_loss(np.array(all_scores["train"]).flatten())
        torch.save({"model": self.models["train"].state_dict()}, self.save_path)

    def test(self):
        scores = self.task_functions["test"](
            self.models["test"], self.dataloaders["test"]
        )
        print(np.mean(scores))

    def plot_loss(self, losses):
        plt.plot(losses)
        plt.xlabel("training iteration")
        plt.ylabel("loss")
        plt.savefig("reports/figures/loss.png")


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    data_root = "data/raw"
    print("Training day and night")
    print(lr)

    tasks = ["train", "test"]
    datasets = {task: MNIST(data_root, task) for task in tasks}
    dataloaders = {
        task: DataLoader(datasets[task], batch_size=16, num_workers=0) for task in tasks
    }
    classifier = Classifier()
    inference = Inference(classifier)
    models = {"train": classifier, "test": inference}

    optimizer = torch.optim.Adam(classifier.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4)
    criterion = torch.nn.CrossEntropyLoss()

    main = Main(
        models, dataloaders, optimizer, scheduler, criterion, "models/trained_model.pt"
    )
    main.train(5)


cli.add_command(train)

if __name__ == "__main__":
    print("i enter train_model")
    cli()
