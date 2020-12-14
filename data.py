from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from transformers import PreTrainedTokenizer

DATA_ROOT = Path('data')


class Vocabulary:

    pad_idx = 0
    pad_token = '<pad>'
    unk_idx = 1
    unk_token = '<unk>'

    def __init__(self, cache_fname: str = 'vocab.csv', train_fname: str = 'senti.train.tsv',
                 max_vocab_size: Optional[int] = None):
        """
        Non-pretrained tokenizer built from training set.

        Args:
            cache_fname: Filename for preprocessed vocabulary in
                DATA_ROOT.
            train_fname: Filename for training data in DATA_ROOT.
            max_vocab_size: If set, caps the vocabulary size at
                max_vocab_size + 2. Default: None
        """
        special = pd.DataFrame({'token': [self.pad_token, self.unk_token], 'freq': [0, 0]})
        cache = DATA_ROOT / cache_fname
        if cache.exists():
            vocab = pd.read_csv(cache)
        else:
            counter = Counter()
            with open(DATA_ROOT / train_fname) as f:
                for line in f:
                    counter += Counter(line.split('\t')[0].lower().split(' '))
            vocab = pd.DataFrame(counter.most_common(max_vocab_size), columns=['token', 'freq'])
            vocab.to_csv(cache, index=False)
        self.df = pd.concat([special, vocab], ignore_index=True)
        self.itos = self.df['token']
        self.stoi = {token: i for i, token in self.itos.items()}

    def __len__(self) -> int:
        return len(self.itos)


class SentiDataset(Dataset):

    def __init__(self, filename: str, vocab: Vocabulary):
        """
        Dataset for non-pretrained models.

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
        Collate function working with SentiDataset.

        Args:
            pad_idx: Padding index. Default: Vocabulary.pad_idx
        """
        self.pad_idx = pad_idx

    def __call__(self, batch: Sequence[Tuple[torch.LongTensor, int]]) -> Tuple[torch.LongTensor, torch.Tensor]:
        sentences, labels = zip(*batch)
        return pad_sequence(sentences, batch_first=True, padding_value=self.pad_idx), torch.Tensor(labels)


class PaddedSentiDataset(Dataset):

    def __init__(self, filename: str, tokenizer: PreTrainedTokenizer):
        """
        Dataset for BERT models.

        Args:
            filename: Dataset filename in DATA_ROOT.
            tokenizer: Pretrained tokenizer.
        """
        self.df = pd.read_csv(DATA_ROOT / filename, sep='\t', names=['text', 'label'])
        self.df['label'] = self.df['label'].astype(float)
        self.encodings = tokenizer(list(self.df['text']), truncation=True, padding=True)['input_ids']

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, float]:
        """

        Args:
            idx: Index.

        Returns:
            Tokenizer encoded text, label
        """
        encoded = self.encodings[idx]
        label = self.df.loc[idx, 'label']
        return torch.LongTensor(encoded), label


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
