from logging import getLogger
from pathlib import Path
from typing import List

import gensim
import torch
from flair.data import Sentence
from torch import Tensor
from torchglyph.proc import Proc, LoadVectors
from torchglyph.vocab import Vectors, Glove

from ner import project_data_dir

logger = getLogger(__name__)

__all__ = [
    'Transpose',
    'ToFlairSentence',
    'PubMed', 'LoadPubMed',
]


class Transpose(Proc):
    def __call__(self, tensor: Tensor, **kwargs) -> Tensor:
        return tensor.transpose(0, 1)


class ToFlairSentence(Proc):
    def __call__(self, xs: List[str], *args, **kwargs) -> Sentence:
        sentence = Sentence()
        for token in xs:
            sentence.add_token(token)
        return sentence


class PubMed(Vectors):
    vector_format = 'word2vec'

    @classmethod
    def paths(cls, root: Path = project_data_dir, **kwargs) -> List[Path]:
        return [root / 'pubmed' / 'PubMed-shuffle-win-2.bin']

    def cache_(self, path: Path) -> None:
        torch_path = path.with_suffix('.pt')

        if not torch_path.exists():
            keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(
                str(path), binary=True, encoding='utf-8',
            )
            for token in keyed_vectors.index2word:
                self.add_token_(token)

            self.vectors = torch.tensor(keyed_vectors.vectors, dtype=torch.float32)
            self.save(torch_path)
        else:
            self.load(torch_path)


class LoadGlove(LoadVectors):
    def __init__(self, *fallbacks, name: str, dim: int) -> None:
        super(LoadGlove, self).__init__(
            *fallbacks, vectors=Glove(name=name, dim=dim, root=project_data_dir),
        )


class LoadPubMed(LoadVectors):
    def __init__(self, *fallback_fns) -> None:
        super(LoadPubMed, self).__init__(
            *fallback_fns,
            vectors=PubMed(root=project_data_dir),
        )
