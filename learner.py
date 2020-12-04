from typing import Callable, Tuple
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from transformers import PreTrainedModel


class SentimentLearner:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
                 loss_fn: Callable[..., torch.Tensor], optim_cls: Callable[..., Optimizer], lr: float):
        """

        Args:
            model: Model.
            train_loader: DataLoader for training dataset.
            valid_loader: DataLoader for validation dataset.
            loss_fn: Loss function.
            optim_cls: Optimizer class.
            lr: Learning rate.
        """
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.loss_fn = loss_fn
        self.optimizer = optim_cls(self.model.parameters(), lr=lr)

    def _get_metrics(self, batch: Tuple[torch.LongTensor, torch.LongTensor]) -> Tuple[torch.Tensor, float, int]:
        """
        Passes a batch of data and returns metrics, such as loss,
        accuracy, etc.

        Args:
            batch: One batch data from iterating DataLoader.

        Returns:
            loss, accuracy, batch_size
        """
        sequences, targets = batch
        sequences, targets = sequences.to(self.device), targets.to(self.device)

        logits = self.model(sequences).logits.unsqueeze() if isinstance(self.model, PreTrainedModel) \
            else self.model(sequences)
        loss = self.loss_fn(logits, targets)
        predictions = torch.sigmoid(logits).round_()
        accuracy = (predictions == targets).sum().item() / len(targets)

        return loss, accuracy, len(targets)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader = None) -> Tuple[float, float]:
        """
        Evaluates metrics with a given DataLoader or just valid_loader.

        Args:
            data_loader: DataLoader, use valid_loader if not provided.
                Default: None

        Returns:
            Loss, accuracy
        """
        self.model.eval()
        if data_loader is None:
            data_loader = self.valid_loader

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
            valid_loss, valid_acc = self.evaluate()
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
