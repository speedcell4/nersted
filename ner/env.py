import logging
import random
import sys
from logging import getLogger
from pathlib import Path

import colorlog
import numpy as np
import torch

from ner import get_out_dir

logger = getLogger(__name__)

__all__ = [
    'setup_env',
]


def setup_logger(out_dir: Path,
                 level: int = logging.INFO,
                 fmt: str = '%(asctime)s [%(levelname)-s] %(name)s | %(message)s') -> None:
    torch.set_printoptions(precision=4, sci_mode=False)

    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
        handler.close()
    logging.root.setLevel(level=level)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(colorlog.ColoredFormatter(
        fmt='%(log_color)s' + fmt,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'bold_red',
            'ERROR': 'bold_orange',
            'CRITICAL': 'bold_purple',
        },
    ))
    handler.setLevel(level=level)
    logging.root.addHandler(hdlr=handler)

    handler = logging.FileHandler(filename=(out_dir / 'log.txt').__str__(), mode='w', encoding='utf-8')
    handler.setFormatter(fmt=logging.Formatter(fmt=fmt))
    handler.setLevel(level=level)
    logging.root.addHandler(hdlr=handler)

    logger.critical(' '.join(sys.argv))


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_env(study: str, seed: int = 42, **kwargs):
    out_dir = get_out_dir(study)
    setup_logger(out_dir=out_dir)
    setup_seed(seed=seed)
    return torch.device('cuda:0'), out_dir
