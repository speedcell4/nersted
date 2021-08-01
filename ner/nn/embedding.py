from typing import Type
from typing import Union

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchglyph.nn import TokenEmbedding, CharLstmEmbedding
from torchglyph.vocab import Vocab

__all__ = [
    'Embedding',
    'EmbeddingLayer',
]


class WordEmbedding(TokenEmbedding):
    def __init__(self, freeze: bool = False, *, vocab: Vocab = None) -> None:
        super(WordEmbedding, self).__init__(
            embedding_dim=vocab.vectors.size()[1], freeze=freeze, vocab=vocab,
        )


class CharEmbedding(TokenEmbedding):
    def __init__(self, embedding_dim: int = 30, freeze: bool = False, *, vocab: Vocab = None) -> None:
        super(CharEmbedding, self).__init__(
            embedding_dim=embedding_dim, freeze=freeze, vocab=vocab,
        )


class EmbeddingLayer(nn.Module):
    def __init__(self, word_: Type[WordEmbedding] = WordEmbedding,
                 char_: Type[CharEmbedding] = CharEmbedding,
                 char_rnn_: Type[CharLstmEmbedding] = CharLstmEmbedding,
                 dropout: float = 0.5,
                 *,
                 word_vocab: Vocab, char_vocab: Vocab, ctx_dim: int) -> None:
        super(EmbeddingLayer, self).__init__()

        self.ctx_dim = ctx_dim
        self.embedding_dim = self.ctx_dim

        self.word_embedding = word_(vocab=word_vocab)
        self.embedding_dim += self.word_embedding.embedding_dim

        self.char_embedding = char_rnn_(char_embedding=char_(vocab=char_vocab))
        self.embedding_dim += self.char_embedding.embedding_dim

        self.dropout = nn.Dropout(dropout)

    def extra_repr(self) -> str:
        args = []
        if self.ctx_dim > 0:
            args.append(f'ctx_dim={self.ctx_dim}')
        return ', '.join(args)

    def forward(self, batch) -> PackedSequence:
        word_embedding = self.word_embedding(batch.word)  # type: PackedSequence
        char_embedding = self.char_embedding(batch.char)  # type: Tensor
        embeddings = [word_embedding.data, char_embedding]

        if self.ctx_dim > 0:
            embeddings.append(batch.ctx.data)

        data = torch.cat(embeddings, dim=-1)  # type: Tensor
        return word_embedding._replace(data=self.dropout(data))


Embedding = Union[
    Type[EmbeddingLayer],
]
