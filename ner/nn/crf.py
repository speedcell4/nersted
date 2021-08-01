from typing import Union, Type

from torchglyph.vocab import Vocab
from torchlatent import crf

__all__ = [
    'CRF',
    'CrfDecoder',
]


class CrfDecoder(crf.CrfDecoder):
    def __init__(self, *, tag_vocab: Vocab) -> None:
        super(CrfDecoder, self).__init__(
            num_tags=len(tag_vocab),
            num_conjugates=1,
        )


CRF = Union[
    Type[CrfDecoder],
]
