from logging import getLogger
from typing import Union, Type

from torch import nn
from torch.nn import init
from torchglyph.vocab import Vocab

logger = getLogger(__name__)

__all__ = [
    'Projector',
    'Linear',
]


class Linear(nn.Linear):
    def __init__(self, *, in_size: int, vocab: Vocab) -> None:
        super(Linear, self).__init__(
            in_features=in_size,
            out_features=len(vocab),
            bias=False,
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=5 ** 0.5, nonlinearity='relu')


Projector = Union[
    Type[Linear],
]
