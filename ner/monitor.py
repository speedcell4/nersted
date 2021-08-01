from logging import getLogger

from torch import nn

logger = getLogger(__name__)

__all__ = [
    'monitor_param_size',
]


def monitor_param_size(model: nn.Module) -> None:
    num_param = sum(param.numel() for param in model.parameters())
    logger.info(f'{model.__class__.__name__} ({num_param}) => {model}')
