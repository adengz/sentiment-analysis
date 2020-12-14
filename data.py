from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict, Union, Tuple, Callable, Sequence

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

    def __call__(self, texts: List[str]) -> Dict[str, List[int]]:
        """
        Tokenizes a sequence of texts.

        Args:
            texts: Sentences to be tokenized.

        Returns:
            Tokenized result, including input_ids and attention_mask.
        """
        tokenized = {'input_ids': [], 'attention_mask': []}
        for sentence in texts:
            encoded = [self.stoi.get(w, self.unk_idx) for w in sentence.lower().split(' ')]
            mask = [1] * len(encoded)
            tokenized['input_ids'].append(encoded)
            tokenized['attention_mask'].append(mask)
        return tokenized


class SentiDataset(Dataset):

    def __init__(self, filename: str, tokenizer: Union[Vocabulary, PreTrainedTokenizer]):
        """
        Dataset for sentiment analysis, regardless of whether model is
        pretrained.

        Args:
            filename: Dataset filename in DATA_ROOT.
            tokenizer: Tokenizer.
        """
        texts, self.labels = [], []
        with open(DATA_ROOT / filename) as f:
            for line in f:
                text, label = line.split('\t')
                texts.append(text)
                self.labels.append(label)
        self.tokenizer = tokenizer
        kwargs = {}
        if isinstance(self.tokenizer, PreTrainedTokenizer):
            kwargs.update(dict(truncation=True, padding=True))
        self.tokenized = self.tokenizer(texts, **kwargs)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor, float]:
        """

        Args:
            idx: Index.

        Returns:
            Encoded text, attention mask, label
        """
        return torch.LongTensor(self.tokenized['input_ids'][idx]), \
            torch.LongTensor(self.tokenized['attention_mask'][idx]), \
            float(self.labels[idx])

    def get_collate_fn(self) \
            -> Optional[Callable[[Sequence[Tuple[torch.LongTensor, torch.LongTensor, float]]],
                                 Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]]]:
        """

        Returns:
            Collate function used for DataLoader.
        """
        if isinstance(self.tokenizer, Vocabulary):
            def padding_collate(batch: Sequence[Tuple[torch.LongTensor, torch.LongTensor, float]]) \
                    -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
                """

                Args:
                    batch: Batch of data from SentiDataset.

                Returns:
                    padded text encodings: batch_size, pad_len
                    padded attention masks: batch_size, pad_len
                    labels: batch_size
                """
                encodings, masks, labels = zip(*batch)
                return pad_sequence(encodings, batch_first=True, padding_value=self.tokenizer.pad_idx), \
                    pad_sequence(masks, batch_first=True, padding_value=0), torch.Tensor(labels)
            return padding_collate


def get_dataloader(dataset: SentiDataset, batch_size: int, shuffle: bool = True, pin_memory: bool = True) \
        -> DataLoader:
    """
    Wrapper function for creating a DataLoader with a given dataset.

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=dataset.get_collate_fn(), pin_memory=pin_memory)
