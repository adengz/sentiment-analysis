from pathlib import Path
from collections import Counter
from typing import Tuple, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

DATA_ROOT = Path('data')


class Vocabulary:

    pad_idx = 0
    pad_token = '<pad>'
    unk_idx = 1
    unk_token = '<unk>'

    def __init__(self, train_fname: str = 'senti.train.tsv', max_vocab_size: int = None):
        """

        Args:
            train_fname: Filename for training data in DATA_ROOT.
            max_vocab_size: If set, caps the vocabulary size at
                max_vocab_size + 2. Default: None
        """
        self.itos = [self.pad_token, self.unk_token]
        self.freqs = [0] * 2
        counter = Counter()
        with open(DATA_ROOT / train_fname) as f:
            for line in f:
                counter += Counter(line.split('\t')[0].lower().split(' '))
        for word, freq in counter.most_common(max_vocab_size):
            self.itos.append(word)
            self.freqs.append(freq)
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)


class SentiDataset(Dataset):

    def __init__(self, filename: str, vocab: Vocabulary):
        """

        Args:
            filename: Dataset filename in DATA_ROOT.
            vocab: Vocabulary.
        """
        self.df = pd.read_csv(DATA_ROOT / filename, sep='\t', names=['text', 'label'])
        self.df['label'] = self.df['label'].astype(float)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, float]:
        """

        Args:
            idx: Index.

        Returns:
            Vocabulary encoded text, label
        """
        sentence, label = self.df.iloc[idx]
        encoded = [self.vocab.stoi.get(w, self.vocab.unk_idx) for w in sentence.lower().split(' ')]
        return torch.LongTensor(encoded), label


class PadSeqCollate:

    def __init__(self, pad_idx: int = Vocabulary.pad_idx):
        """

        Args:
            pad_idx: Padding index. Default: Vocabulary.pad_idx
        """
        self.pad_idx = pad_idx

    def __call__(self, batch: Sequence[Tuple[torch.LongTensor, int]]) -> Tuple[torch.LongTensor, torch.Tensor]:
        sentences, labels = zip(*batch)
        return pad_sequence(sentences, padding_value=self.pad_idx), torch.Tensor(labels)


def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, pin_memory: bool = True) \
        -> DataLoader:
    """
    Wrapper function for creating a DataLoader with a given dataset.

    Args:
        dataset: Dataset.
        batch_size: Batch size.
        shuffle: Whether reshuffle data at each epoch, see DataLoader
            docs. Default: True
        pin_memory: Whether use pinned memory, see DataLoader docs.
            Default: True

    Returns:
        DataLoader.
    """
    collate_fn = PadSeqCollate() if isinstance(dataset, SentiDataset) else None
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_fn, pin_memory=pin_memory)
