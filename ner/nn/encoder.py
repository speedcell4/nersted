from typing import Type, Union

from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchglyph.vocab import Vocab
from torchrua import PackedSequential

from ner.nn.projector import Linear

__all__ = [
    'Encoder',
    'LstmEncoder',
]


class LstmEncoderLayer(nn.Module):
    def __init__(self, dropout: float = 0.5, *,
                 in_size: int, chunk_size: int, num_chunks: int) -> None:
        super(LstmEncoderLayer, self).__init__()

        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=num_chunks * chunk_size,
            bidirectional=True, bias=True,
        )
        self.encoding_dim = self.rnn.hidden_size
        if self.rnn.bidirectional:
            self.encoding_dim *= 2

        self.layer_norm = PackedSequential(
            Rearrange('... (d c x) -> ... c (d x)', d=2, c=num_chunks),
            nn.LayerNorm(chunk_size * 2),
            nn.Dropout(dropout),
            Rearrange('... c (d x) -> ... (d c x)', d=2, c=num_chunks),
        )

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        encoding, _ = self.rnn(sequence)
        return self.layer_norm(encoding)


class LstmProjectorLayer(nn.Module):
    def __init__(self, dropout: float = 0.5,
                 proj: Type[Linear] = Linear, *,
                 in_size: int, chunk_size: int, num_chunks: int, tag_vocab: Vocab) -> None:
        super(LstmProjectorLayer, self).__init__()

        self.rnn = nn.LSTM(
            input_size=in_size,
            hidden_size=num_chunks * chunk_size,
            bidirectional=True, bias=True,
        )
        self.encoding_dim = self.rnn.hidden_size
        if self.rnn.bidirectional:
            self.encoding_dim *= 2

        self.projector = PackedSequential(
            Rearrange('... (d c x) -> ... c (d x)', d=2, c=num_chunks),
            nn.Dropout(dropout),
            proj(in_size=chunk_size * 2, vocab=tag_vocab),
        )

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        encoding, _ = self.rnn(sequence)
        return self.projector(encoding)


class LstmEncoder(nn.Sequential):
    def __init__(self, enc_layer: Type[LstmEncoderLayer] = LstmEncoderLayer,
                 proj_: Type[LstmProjectorLayer] = LstmProjectorLayer,
                 num_layers: int = 3, chunk_size: int = 50, *,
                 in_size: int, num_chunks: int, tag_vocab: Vocab) -> None:
        args = []
        for _ in range(1, num_layers):
            layer = enc_layer(
                in_size=in_size, chunk_size=chunk_size,
                num_chunks=num_chunks,
            )
            args.append(layer)

            in_size = layer.encoding_dim

        layer = proj_(
            in_size=in_size, chunk_size=chunk_size,
            num_chunks=num_chunks, tag_vocab=tag_vocab,
        )
        args.append(layer)

        super(LstmEncoder, self).__init__(*args)


Encoder = Union[
    Type[LstmEncoder],
]
