from logging import getLogger
from typing import Type, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = getLogger(__name__)

__all__ = [
    'Scheduler',
    'Gamma',
]


class Gamma(LambdaLR):
    def __init__(self, gamma: float = 0.05, *, optimizer: Optimizer, **kwargs) -> None:
        self.gamma = gamma

        def lr_lambda(epoch: int) -> float:
            return 1 / (1 + gamma * epoch)

        super(Gamma, self).__init__(optimizer=optimizer, lr_lambda=lr_lambda)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(gamma={self.gamma})'

    def step_update(self, **kwargs) -> None:
        pass

    def epoch_update(self, **kwargs) -> None:
        self.step(**kwargs)
        lr = ', '.join([f'{lr:.6f}' for lr in self.get_last_lr()])
        logger.info(f'learning rate => [{lr}]')


Scheduler = Union[
    Type[Gamma],
]
