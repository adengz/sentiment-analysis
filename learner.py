import time
from typing import Callable, Tuple, Dict

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

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
        self.train_loader = get_dataloader(train_set, batch_size=batch_size)
        self.valid_loader = get_dataloader(valid_set, batch_size=batch_size)
        self.test_loader = get_dataloader(test_set, batch_size=batch_size)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim_cls(self.model.parameters(), lr=lr)

    def _get_metrics(self, batch: Tuple[Dict[str, torch.LongTensor], torch.Tensor]) \
            -> Tuple[torch.Tensor, float, int]:
        """
        Passes a batch of data and returns metrics, such as loss,
        accuracy, etc.

        Args:
            batch: One batch data from iterating DataLoader.

        Returns:
            Loss, accuracy, batch size
        """
        tokenized, labels = batch
        for k in tokenized:
            tokenized[k] = tokenized[k].to(self.device)
        labels = labels.to(self.device)

        logits = self.model(**tokenized).logits.squeeze()
        loss = self.loss_fn(logits, labels)
        predictions = torch.sigmoid(logits).round_()
        accuracy = (predictions == labels).float().mean().item()

        return loss, accuracy, len(labels)

    @torch.no_grad()
    def evaluate(self, valid: bool = False) -> Tuple[float, float]:
        """
        Evaluates metrics with validation or testing dataset.

        Args:
            valid: Use valid dataset (True) or test dataset (False).
                Default: False

        Returns:
            Loss, accuracy
        """
        self.model.eval()
        data_loader = self.valid_loader if valid else self.test_loader

        sum_loss = sum_acc = total_count = 0
        for batch in data_loader:
            loss, acc, count = self._get_metrics(batch)

            total_count += count
            sum_loss += loss.item() * count
            sum_acc += acc * count

        return sum_loss / total_count, sum_acc / total_count

    def _train_1_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Trains model by 1 epoch.

        Args:
            epoch: Current epoch.

        Returns:
            Loss, accuracy
        """
        self.model.train()
        loader = tqdm(self.train_loader, desc=f'Epoch {epoch + 1:02}', total=len(self.train_loader))

        sum_loss = sum_acc = total_count = 0
        for batch in loader:
            self.optimizer.zero_grad()
            loss, acc, count = self._get_metrics(batch)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_count += count
            batch_loss = loss.item()
            sum_loss += batch_loss * count
            sum_acc += acc * count
            loader.set_postfix({'Loss': batch_loss, 'Acc': acc})

        return sum_loss / total_count, sum_acc / total_count

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
            train_loss, train_acc = self._train_1_epoch(epoch)
            valid_loss, valid_acc = self.evaluate(valid=True)

            print(f'\tTrain Loss: {train_loss:.3f}\tTrain Acc: {train_acc * 100:.2f}%')
            print(f'\tValid Loss: {valid_loss:.3f}\tValid Acc: {valid_acc * 100:.2f}%')

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                torch.save(self.model.state_dict(), filename)
                print(f'\tModel parameters saved to {filename}')

            time.sleep(0.5)  # avoid nested tqdm chaos

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

    def print_test_results(self):
        """
        Prints testing loss and accuracy.
        """
        test_loss, test_acc = self.evaluate()
        print(f'\t Test Loss: {test_loss:.3f}\t Test Acc: {test_acc * 100:.2f}%')
