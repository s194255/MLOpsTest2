from src.models.model import Classifier
import torch
import pytest

def can_calculate_loss(model, criterion, batch_size):
    x = torch.randn((batch_size, 1, 28, 28), dtype=torch.float32)
    labels = torch.randint(10, (batch_size,), dtype=torch.int64)
    preds = model(x)
    try:
        loss = criterion(preds, labels)
        assert loss.dtype == torch.float32, 'loss did not have torch.float32 as dtype'
        return True
    except:
        return False

@pytest.mark.parametrize('batch_size', [1, 2, 4, 8, 16, 32])
def test_training(batch_size):
    criterion = torch.nn.CrossEntropyLoss()
    model = Classifier()
    can_calculate_loss(model, criterion, batch_size)
    # for batch_size in [1, 2, 4, 8, 16, 32, 64]:
    #     assert can_calculate_loss(model, criterion, batch_size), 'loss could not be calculated'
