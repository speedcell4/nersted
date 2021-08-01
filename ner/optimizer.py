from logging import getLogger
from typing import Type, Union

from torch import nn, optim

__all__ = [
    'Optimizer',
    'SGD',
]

logger = getLogger(__name__)


class SGD(optim.SGD):
    def __init__(self, lr: float = 0.05, weight_decay: float = 1e-8,
                 momentum: float = 0.9, nesterov: bool = False, *, model: nn.Module) -> None:
        super(SGD, self).__init__(
            params=model.parameters(), weight_decay=weight_decay,
            lr=lr, momentum=momentum, nesterov=nesterov,
        )


Optimizer = Union[
    Type[SGD],
]
