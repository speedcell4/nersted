import re
from typing import List

import torch
from seqeval.metrics.sequence_labeling import get_entities
from torch.nn.utils.rnn import PackedSequence
from torchglyph.pipe import PackListStrPipe, PackListListStrPipe
from torchglyph.pipe import Pipe
from torchglyph.proc import UpdateCounter, Numbering, Lift, ToTensor, ToDevice, PackSequences, RegexSub
from torchrua import pad_packed_sequence

from ner.data.proc import Transpose
from ner.data.utils import Entity

__all__ = [
    'WordPipe', 'CharPipe', 'CtxPipe', 'TagPipe',
]


class WordPipe(PackListStrPipe):
    def __init__(self, device: torch.device) -> None:
        super(WordPipe, self).__init__(
            device=device, unk_token='<unk>', special_tokens=(),
            threshold=10, dtype=torch.long,
        )
        self.with_(
            pre=RegexSub(pattern=re.compile(r'\d+'), repl='<digits>') + ...,
        )


class CharPipe(PackListListStrPipe):
    def __init__(self, device: torch.device) -> None:
        super(CharPipe, self).__init__(
            device=device, unk_token='<unk>', special_tokens=(),
            threshold=10, dtype=torch.long,
        )


class CtxPipe(Pipe):
    def __init__(self, device: torch.device) -> None:
        super(CtxPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=torch.float32) + ToDevice(device=device),
            batch=PackSequences(device=device),
        )


class TagPipe(PackListStrPipe):
    def __init__(self, device: torch.device) -> None:
        super(TagPipe, self).__init__(
            device=device, unk_token='O', special_tokens=(),
            threshold=1000, dtype=torch.long,
        )
        self.with_(
            pre=Lift(UpdateCounter()),
            post=Numbering() + ToTensor(dtype=torch.long) + Transpose() + ToDevice(device=device),
        )

    @staticmethod
    def inv_num(batch: PackedSequence) -> List[List[List[int]]]:
        data, lengths = pad_packed_sequence(batch, batch_first=True)
        data = data.detach().cpu()
        lengths = lengths.detach().cpu().tolist()

        sequence = []
        for index1, length in enumerate(lengths):
            sequence.append([])

            for index3 in range(data.size()[-1]):
                sequence[-1].append([
                    data[index1, index2, index3].item()
                    for index2 in range(length)
                ])

        return sequence

    def inv(self, batch: PackedSequence) -> List[List[List[Entity]]]:
        vocab = self.vocab.itos
        return [
            [get_entities([vocab[token] for token in sequence]) for sequence in chunks]
            for chunks in self.inv_num(batch=batch)
        ]
