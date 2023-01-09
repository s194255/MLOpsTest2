import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, hidden_dim=50):
        super().__init__()

        self.feature_extractor = FeatureExtractor()
        self.head = Head(96, 10, hidden_dim=hidden_dim)

    def forward(self, x):
        self._check_input(x)
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        return x

    def _check_input(self, x):
        # if x.shape != torch.Size([x.shape[0], 1, 28, 28]):
        #     raise ValueError('Expect a tensor of shape (batch_size, 1, 28, 28)')
        if len(x.shape) != 4:
            raise ValueError("Expected 4D tensor")
        else:
            if x.shape[1] != 1:
                raise ValueError("Expected 1 channel dim as input")
            if x.shape[2:] != torch.Size([28, 28]):
                raise ValueError("Expected spatial dimension to be 28x28")
        if x.dtype != torch.float32:
            raise ValueError("Excepts dtype torch.float32 but got {}".format(x.dtype))


class Inference(nn.Module):
    def __init__(self, classifier):
        super().__init__()

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        x = torch.argmax(x, dim=1)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        self.block = nn.Sequential(
            ConvBlock(1, 12),
            ConvBlock(12, 12),
            nn.MaxPool2d(3),
            ConvBlock(12, 24),
            nn.MaxPool2d(3),
        )

    def forward(self, x):
        return self.block(x)


class Head(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=50):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
