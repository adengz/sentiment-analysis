from typing import Callable, Tuple
import time

import torch
from torch import nn
from torch.optim import Optimizer

from data import SentiDataset, get_dataloader


class SentimentLearner:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, model: nn.Module, train_set: SentiDataset, valid_set: SentiDataset, test_set: SentiDataset,
                 batch_size: int, optim_cls: Callable[..., Optimizer], lr: float):
        """
        Learner for training binary sentiment analysis models.

        Args:
            model: Model.
            train_set: Training dataset.
            valid_set: Validation dataset.
            test_set: Testing dataset.
            batch_size: Batch size.
            optim_cls: Optimizer class.
            lr: Learning rate.
        """
        self.model = model.to(self.device)
        self.train_loader = get_dataloader(train_set, batch_size)
        self.valid_loader = get_dataloader(valid_set, batch_size)
        self.test_loader = get_dataloader(test_set, batch_size)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim_cls(self.model.parameters(), lr=lr)

    def _get_metrics(self, batch: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]) \
            -> Tuple[torch.Tensor, float, int]:
        """
        Passes a batch of data and returns metrics, such as loss,
        accuracy, etc.

        Args:
            batch: One batch data from iterating DataLoader.

        Returns:
            loss, accuracy, batch_size
        """
        encodings, masks, targets = batch
        encodings, masks, targets = encodings.to(self.device), masks.to(self.device), targets.to(self.device)

        logits = self.model(encodings, masks).logits.squeeze()
        loss = self.loss_fn(logits, targets)
        predictions = torch.sigmoid(logits).round_()
        accuracy = (predictions == targets).float().mean().item()

        return loss, accuracy, len(targets)

    @torch.no_grad()
    def evaluate(self, valid: bool = False) -> Tuple[float, float]:
        """
        Evaluates metrics with a given DataLoader or just valid_loader.

        Args:
            valid: Use valid dataset (True) or test dataset (False).
                Default: False

        Returns:
            Loss, accuracy
        """
        self.model.eval()
        data_loader = self.valid_loader if valid else self.test_loader

        epoch_loss = epoch_acc = total_count = 0
        for batch in data_loader:
            loss, acc, count = self._get_metrics(batch)

            total_count += count
            epoch_loss += loss.item() * count
            epoch_acc += acc * count

        return epoch_loss / total_count, epoch_acc / total_count

    def _train_1_epoch(self) -> Tuple[float, float]:
        """
        Trains model by 1 epoch.

        Returns:
            Loss, accuracy
        """
        self.model.train()

        epoch_loss = epoch_acc = total_count = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss, acc, count = self._get_metrics(batch)

            loss.backward()
            self.optimizer.step()

            total_count += count
            epoch_loss += loss.item() * count
            epoch_acc += acc * count

        return epoch_loss / total_count, epoch_acc / total_count

    def train(self, epochs: int, filename: str):
        """
        Trains model by multiple epochs and saves the parameters of
        the model with the lowest validation loss.

        Args:
            epochs: Number of epochs to train.
            filename: Filename to save model parameters.
        """
        min_valid_loss = float('inf')

        for epoch in range(epochs):
            start = time.time()
            train_loss, train_acc = self._train_1_epoch()
            valid_loss, valid_acc = self.evaluate(valid=True)
            end = time.time()

            print(f'Epoch : {epoch + 1:02}\tWall time : {end - start:.3f}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}%')

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'{filename}')
                print(f'\tModel parameters saved to {filename}')
            else:
                print()

    def load_model_params(self, filename: str):
        """
        Loads parameters from a file. Do nothing if parameters not
        matching exactly.

        Args:
            filename: Filename with saved model parameters.
        """
        curr_state = self.model.state_dict()
        missing_keys, unexpected_keys = self.model.load_state_dict(torch.load(filename))
        if missing_keys or unexpected_keys:
            self.model.load_state_dict(curr_state)
            raise KeyError('Parameters not matching with model, aborted.')
