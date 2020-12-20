from pathlib import Path
from collections import Counter
from functools import partial
from typing import Optional, List, Dict, Union, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from transformers import PreTrainedTokenizerFast

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

    def __call__(self, text: List[str]) -> Dict[str, List[List[int]]]:
        """
        Tokenizes a group of sentences.

        Args:
            text: Sentences to be tokenized.

        Returns:
            Tokenized results, input_ids only.
        """
        input_ids = []
        for sentence in text:
            encoded = [self.stoi.get(w, self.unk_idx) for w in sentence.lower().split(' ')]
            input_ids.append(encoded)

        return {'input_ids': input_ids}


class SentiDataset(Dataset):

    def __init__(self, filename: str, tokenizer: Union[Vocabulary, PreTrainedTokenizerFast]):
        """
        Dataset for sentiment analysis.

        Args:
            filename: Dataset filename in DATA_ROOT.
            tokenizer: Tokenizer.
        """
        texts, self.labels = [], []
        with open(DATA_ROOT / filename) as f:
            for line in f:
                text, label = line.split('\t')
                texts.append(text)
                self.labels.append(int(label))
        self.encodings = tokenizer(texts)['input_ids']

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, int]:
        """

        Args:
            idx: Index.

        Returns:
            Encoded sentence, label
        """
        return torch.LongTensor(self.encodings[idx]), self.labels[idx]


pad_zeros = partial(pad_sequence, batch_first=True, padding_value=0)


def padding_collate(batch: List[Tuple[torch.LongTensor, int]]) -> Tuple[Dict[str, torch.LongTensor], torch.Tensor]:
    """
    Collate function bridging SentiDataset and all models.

    Args:
        batch: Batch of data from SentiDataset.

    Returns:
        Batched tokenized results (input_ids & attention_mask),
        batched labels
    """
    input_ids, labels = zip(*batch)
    attention_masks = list(map(lambda encoded: torch.ones_like(encoded), input_ids))
    return {'input_ids': pad_zeros(input_ids), 'attention_mask': pad_zeros(attention_masks)}, torch.Tensor(labels)


def get_dataloader(dataset: SentiDataset, batch_size: int, shuffle: bool = True, pin_memory: bool = True) -> DataLoader:
    """
    Wrapper function for creating a DataLoader loading a SentiDataset.

    Args:
        dataset: SentiDataset.
        batch_size: Batch size.
        shuffle: Whether reshuffle data at each epoch, see DataLoader
            docs. Default: True
        pin_memory: Whether use pinned memory, see DataLoader docs.
            Default: True

    Returns:
        DataLoader.
    """
    return DataLoader(dataset, batch_size=batch_size, collate_fn=padding_collate,
                      shuffle=shuffle, pin_memory=pin_memory)
