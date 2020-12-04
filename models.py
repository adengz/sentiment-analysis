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
            inp: batch_size, seq_len

        Returns:
            batch_size
        """
        embedded = self.embed_dropout(self.embedding(inp))  # batch_size, seq_len, embed_dim
        mean = embedded.mean(1)  # batch_size, embed_dim
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
            inp: batch_size, seq_len

        Returns:
            batch_size
        """
        embedded = self.embed_dropout(self.embedding(inp))  # batch_size, seq_len, embed_dim
        attention = self.attention(embedded).unsqueeze(2)  # batch_size, seq_len, 1
        hidden = torch.mul(embedded, attention).sum(1)  # batch_size, embed_dim
        if self.res_conn:
            hidden += embedded.mean(1)
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
            embedded: batch_size, seq_len, embed_dim

        Returns:
            batch_size, seq_len
        """
        cosine = self.cosine_similarity_to_u(embedded)  # batch_size, seq_len
        attention = F.softmax(cosine, dim=1)  # batch_size, seq_len
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
        embedded: batch_size, seq_len, embed_dim

    Returns:
        batch_size, seq_len
    """
    summed_dot_prod = torch.bmm(embedded, embedded.transpose(1, 2)).sum(2)  # batch_size, seq_len
    attention = F.softmax(summed_dot_prod, dim=1)  # batch_size, seq_len
    return attention


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, model_dim: int, num_heads: int = 1):
        """
        Multi-head self (identical input Q, K, V) attention.

        Args:
            model_dim: d_model.
            num_heads: h.
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert model_dim % num_heads == 0,\
            f'Model dimension {model_dim} not divisible by number of heads {num_heads}'
        head_dim = model_dim // num_heads

        self.to_query = nn.Linear(model_dim, head_dim * num_heads, bias=False)
        self.to_key = nn.Linear(model_dim, head_dim * num_heads, bias=False)
        self.to_value = nn.Linear(model_dim, head_dim * num_heads, bias=False)
        self.fc = nn.Linear(head_dim * num_heads, model_dim)

        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
        """

        Args:
            embedded: batch_size, seq_len, model_dim

        Returns:
            batch_size, seq_len, model_dim
        """
        batch_size = embedded.shape[0]

        # batch_size * num_heads, seq_len, head_dim
        query = self.to_query(embedded).view(batch_size * self.num_heads, -1, self.head_dim)
        key = self.to_key(embedded).view(batch_size * self.num_heads, -1, self.head_dim)
        value = self.to_value(embedded).view(batch_size * self.num_heads, -1, self.head_dim)

        # batch_size * num_heads, seq_len, seq_len
        scaled_dot_prod = torch.bmm(query, key.transpose(1, 2)) / self.head_dim ** 0.5
        attention = F.softmax(scaled_dot_prod, dim=2)

        attended = torch.bmm(attention, value)  # batch_size * num_heads, seq_len, head_dim
        # batch_size, seq_len, head_dim * num_head
        concated = attended.view(batch_size, -1, self.head_dim * self.num_heads)
        output = self.fc(concated)  # batch_size, seq_len, model_dim
        return output


class MultiHeadSelfAttentionModel(nn.Module):

    def __init__(self, vocab_size: int, model_dim: int, num_heads: int = 1, pos_encode: bool = False,
                 embed_dropout: float = 0.5, pad_idx: int = Vocabulary.pad_idx):
        """
        Model implementing multi-head self attention, from input to
        the output of the first transformer encoder sublayer, then
        mapped to a single logit for binary classification.

        Args:
            vocab_size: Vocabulary size.
            model_dim: Word embedding dimension.
            num_heads: Number of attention heads. Default: 1
            pos_encode: Whether add positional encoding on embedded
                sequence. Default: False
            embed_dropout: Dropout applied on word embedding.
                Default: 0.5
            pad_idx: Index of padding token in vocabulary.
                Default: Vocabulary.pad_idx
        """
        super(MultiHeadSelfAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.pos_encode = pos_encode
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.multihead_attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, 1)
        self.model_dim = model_dim

        init_range = 0.5 / model_dim
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.embedding.weight.data[pad_idx].zero_()

    def forward(self, inp: torch.LongTensor) -> torch.Tensor:
        """

        Args:
            inp: batch_size, seq_len

        Returns:
            batch_size
        """
        # batch_size, seq_len, model_dim
        embedded = self.embedding(inp)
        if self.pos_encode:
            embedded += self.get_positional_encoding(inp.shape[1])
        embedded = self.embed_dropout(embedded)

        attention = self.multihead_attention(embedded)
        hidden = self.layer_norm(embedded + attention)  # batch_size, seq_len, model_dim
        logit = self.fc(hidden.mean(1)).squeeze()  # batch_size
        return logit

    def get_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Calculates positional encoding.

        Args:
            seq_len: Input sequence length.

        Returns:
            1, seq_len, model_dim
        """
        pos = torch.arange(seq_len).float()
        dim = torch.arange(self.model_dim)
        frequency = 1 / 10000 ** (dim / self.model_dim)
        pe = torch.matmul(pos[:, None], frequency[None, :])  # seq_len, model_dim
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        pe /= self.model_dim  # to the same scale with embedding

        w = next(self.parameters())  # move pe to the same device as model
        pe = w.new_tensor(pe.tolist())
        return pe[None, :, :]  # 1, seq_len, model_dim
