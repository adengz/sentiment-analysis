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
        return self.embedding.weight.data.cpu()


class AttentionWeightedWordAveragingModel(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, embed_dropout: float = 0.5,
                 pad_idx: int = Vocabulary.pad_idx):
        """
        Adding attention weights on top of word averaging model.

        Args:
            vocab_size: Vocabulary size.
            embed_dim: Word embedding dimension.
            embed_dropout: Dropout applied on word embedding.
                Default: 0.5
            pad_idx: Index of padding token in vocabulary.
                Default: Vocabulary.pad_idx
        """
        super(AttentionWeightedWordAveragingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.fc = nn.Linear(embed_dim, 1)

        init_range = 0.5 / embed_dim
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.embedding.weight.data[pad_idx].zero_()

        u_tensor = torch.Tensor(embed_dim)
        u_tensor.uniform_(-init_range, init_range)
        self.u = nn.Parameter(u_tensor)

    def forward(self, inp: torch.LongTensor, return_attention: bool = False):
        """

        Args:
            inp: seq_len, batch_size
            return_attention: Whether outputs attention instead of
                logit, used only when not evaluating loss.
                    Default: False

        Returns:
            logit: batch_size
            or attention (seq_len, batch_size)
        """
        embedded = self.embed_dropout(self.embedding(inp))  # seq_len, batch_size, embed_dim
        cosine = F.cosine_similarity(embedded, self.u, dim=-1)  # seq_len, batch_size
        attention = F.softmax(cosine, dim=0)  # seq_len, batch_size
        hidden = torch.mul(embedded, attention.unsqueeze(2))  # seq_len, batch_size, embed_dim
        logit = self.fc(hidden.sum(dim=0)).squeeze()  # batch_size
        if return_attention:
            return attention
        return logit

    @property
    def cosine_similarity_to_u(self):
        """
        Cosine similarity between u and embedded word vectors.

        Returns:
            vocab_size
        """
        return F.cosine_similarity(self.embedding.weight.data, self.u, dim=-1).detach().cpu()
