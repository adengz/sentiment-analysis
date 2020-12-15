from pathlib import Path
from collections import Counter
from typing import Optional, List, Dict, Union, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
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

    def __call__(self, text: List[str], padding: bool = False, return_tensors: Optional[str] = None) \
            -> Dict[str, Union[List[List[int]], torch.LongTensor]]:
        """
        Tokenizes a group of sentences.

        Args:
            text: Sentences to be tokenized.
            padding: Whether pads to the longest sentence. Must set to
                True if return_tensors is set. Default: False
            return_tensors: Returns torch.LongTensor instead of
                List[int] if set to 'pt'. Default: None

        Returns:
            Tokenized result, including input_ids and attention_mask.
        """
        input_ids, attention_mask = [], []
        max_len = 0
        for sentence in text:
            encoded = [self.stoi.get(w, self.unk_idx) for w in sentence.lower().split(' ')]
            mask = [1] * len(encoded)
            if len(encoded) > max_len:
                max_len = len(encoded)
            input_ids.append(encoded)
            attention_mask.append(mask)

        if padding:
            for encoded, mask in zip(input_ids, attention_mask):
                pads = max_len - len(encoded)
                encoded += [self.pad_idx] * pads
                mask += [0] * pads

        if return_tensors is None:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif return_tensors == 'pt':
            try:
                return {'input_ids': torch.LongTensor(input_ids), 'attention_mask': torch.LongTensor(attention_mask)}
            except ValueError:
                raise ValueError('Unable to create tensor without padding batch sentences to the same length. '
                                 'Set padding=True.')
        else:
            raise NotImplementedError('Only torch.LongTensor or nested integer list is supported for now.')


class SentiDataset(Dataset):

    def __init__(self, filename: str):
        """
        Pre-tokenized dataset.

        Args:
            filename: Dataset filename in DATA_ROOT.
        """
        self.df = pd.read_csv(DATA_ROOT / filename, sep='\t', names=['text', 'label'])
        self.df['label'] = self.df['label'].astype(float)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, float]:
        """

        Args:
            idx: Index.

        Returns:
            text, label
        """
        return tuple(self.df.iloc[idx])


class TokenizeCollate:

    def __init__(self, tokenizer: Union[Vocabulary, PreTrainedTokenizerFast]):
        """
        Collate function with given tokenizer.

        Args:
            tokenizer: Tokenizer.
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[str, float]]) -> Tuple[Dict[str, torch.LongTensor], torch.Tensor]:
        texts, labels = zip(*batch)
        return self.tokenizer(list(texts), padding=True, return_tensors='pt'), torch.Tensor(labels)


def get_dataloader(dataset: SentiDataset, tokenizer: Union[Vocabulary, PreTrainedTokenizerFast], batch_size: int,
                   shuffle: bool = True, pin_memory: bool = True) -> DataLoader:
    """
    Wrapper function for creating a DataLoader with a given pair of
    dataset and tokenizer.

    Args:
        dataset: SentiDataset.
        tokenizer: Tokenizer.
        batch_size: Batch size.
        shuffle: Whether reshuffle data at each epoch, see DataLoader
            docs. Default: True
        pin_memory: Whether use pinned memory, see DataLoader docs.
            Default: True

    Returns:
        DataLoader.
    """
    return DataLoader(dataset, collate_fn=TokenizeCollate(tokenizer), batch_size=batch_size,
                      shuffle=shuffle, pin_memory=pin_memory)
