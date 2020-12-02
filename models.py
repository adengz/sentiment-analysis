from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from data import Vocabulary


class WordAveragingModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, embed_dropout: float = 0.5,
                 pad_idx: int = Vocabulary.pad_idx):
        """
        (Embedded) word averaging model.

        Args:
            vocab_size: Vocabulary size.
            embed_dim: Word embedding dimension.
            embed_dropout: Dropout applied on word embedding.
                Default: 0.5
            pad_idx: Index of padding token in vocabulary.
                Default: Vocabulary.pad_idx
        """
        super(WordAveragingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.fc = nn.Linear(embed_dim, 1)

        init_range = 0.5 / embed_dim
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.embedding.weight.data[pad_idx].zero_()

    def forward(self, inp: torch.LongTensor) -> torch.Tensor:
        """

        Args:
            inp: seq_len, batch_size

        Returns:
            batch_size
        """
        embedded = self.embed_dropout(self.embedding(inp))  # seq_len, batch_size, embed_dim
        mean = embedded.mean(0)  # batch_size, embed_dim
        logit = self.fc(mean).squeeze()  # batch_size
        return logit

    @property
    def word_embedding(self) -> torch.Tensor:
        """
        Embedded word vectors.

        Returns:
            vocab_size, embed_dim
        """
        return self.embedding.weight.data


class AttentionWeightedWordAveragingModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, attention: Callable[[torch.Tensor], torch.Tensor],
                 res_conn: bool = False, embed_dropout: float = 0.5, pad_idx: int = Vocabulary.pad_idx):
        """
        Adding attention weights on top of word averaging model.

        Args:
            vocab_size: Vocabulary size.
            embed_dim: Word embedding dimension.
            attention: Attention calculator.
            res_conn: Whether apply residual connection to weighted
                hidden state with average embedding. Default: False
            embed_dropout: Dropout applied on word embedding.
                Default: 0.5
            pad_idx: Index of padding token in vocabulary.
                Default: Vocabulary.pad_idx
        """
        super(AttentionWeightedWordAveragingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.attention = attention
        self.fc = nn.Linear(embed_dim, 1)
        self.res_conn = res_conn

        init_range = 0.5 / embed_dim
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.embedding.weight.data[pad_idx].zero_()

    def forward(self, inp: torch.LongTensor) -> torch.Tensor:
        """

        Args:
            inp: seq_len, batch_size

        Returns:
            batch_size
        """
        embedded = self.embed_dropout(self.embedding(inp))  # seq_len, batch_size, embed_dim
        attention = self.attention(embedded).unsqueeze(2)  # seq_len, batch_size, 1
        hidden = torch.mul(embedded, attention).sum(0)  # batch_size, embed_dim
        if self.res_conn:
            hidden += embedded.mean(0)
        logit = self.fc(hidden).squeeze()  # batch_size
        return logit


class UAttention(nn.Module):

    def __init__(self, embed_dim: int):
        """
        Attention computed from cosine similarity between embedded word
        vector and u.

        Args:
            embed_dim: Word embedding dimension.
        """
        super(UAttention, self).__init__()
        init_range = 0.5 / embed_dim
        u_tensor = torch.Tensor(embed_dim)
        u_tensor.uniform_(-init_range, init_range)
        self.u = nn.Parameter(u_tensor)

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """

        Args:
            embedded: seq_len, batch_size, embed_dim

        Returns:
            seq_len, batch_size
        """
        cosine = self.cosine_similarity_to_u(embedded)  # seq_len, batch_size
        attention = F.softmax(cosine, dim=0)  # seq_len, batch_size
        return attention

    def cosine_similarity_to_u(self, embedded: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine similarity between embedded word vector and u.

        Args:
            embedded: *, embed_dim

        Returns:
            *
        """
        return F.cosine_similarity(embedded, self.u, dim=-1)


def dot_product_self_attention(embedded: torch.Tensor) -> torch.Tensor:
    """
    Self attention computed from dot product with other embedded word
    vectors in the same sequence.

    Args:
        embedded: seq_len, batch_size, embed_dim

    Returns:
        seq_len, batch_size
    """
    permuted = embedded.permute(1, 0, 2)  # batch_size, seq_len. embed_dim
    summed_dot_prod = torch.bmm(permuted, permuted.transpose(1, 2)).sum(2)  # batch_size, seq_len
    attention = F.softmax(summed_dot_prod.transpose(0, 1), dim=0)  # seq_len, batch_size
    return attention
