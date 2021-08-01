from logging import getLogger
from typing import List

import torch
from flair.data import Sentence, Token
from flair.embeddings import TransformerWordEmbeddings, StackedEmbeddings, FlairEmbeddings
from torch import Tensor
from torchglyph.proc import Proc

from ner import project_data_dir

logger = getLogger(__name__)

__all__ = [
    'Embedder', 'BERT_DIM', 'FLAIR_DIM',
    'BertEmbedder', 'BioBertEmbedder',
    'FlairNewsEmbedder', 'FlairPubmedEmbedder',
]

BERT_DIM = 1024
FLAIR_DIM = 4096


class Embedder(Proc):
    def __init__(self) -> None:
        super(Embedder, self).__init__()
        logger.info(f'utilizing {self.__class__.__name__}')

    def __call__(self, batch: List[List[str]], **kwargs) -> List[Tensor]:
        sentences = []
        for tokens in batch:
            sentences.append(Sentence())
            for token in tokens:
                sentences[-1].add_token(Token(token))

        self.embedding.embed(sentences)
        with torch.no_grad():
            return [
                torch.stack([
                    token.embedding for token in sentence
                ], dim=0).detach().cpu()
                for sentence in sentences
            ]


class BertEmbedder(Embedder):
    def __init__(self):
        super(BertEmbedder, self).__init__()
        self.embedding = TransformerWordEmbeddings(
            'bert-large-uncased',
            layers='-1,-2,-3,-4', layer_mean=True,
            subtoken_pooling='mean',
            fine_tune=False,
        )


class BioBertEmbedder(Embedder):
    def __init__(self):
        super(BioBertEmbedder, self).__init__()
        self.embedding = TransformerWordEmbeddings(
            f'/{project_data_dir.resolve()}/biobert_large',
            layers='-1,-2,-3,-4', layer_mean=True,
            subtoken_pooling='mean',
            fine_tune=False, from_tf=True,
        )
        self.embedding.tokenizer.basic_tokenizer.do_lower_case = False


class FlairNewsEmbedder(Embedder):
    def __init__(self):
        super(FlairNewsEmbedder, self).__init__()
        self.embedding = StackedEmbeddings([
            FlairEmbeddings('news-forward', fine_tune=False),
            FlairEmbeddings('news-backward', fine_tune=False),
        ])


class FlairPubmedEmbedder(Embedder):
    def __init__(self):
        super(FlairPubmedEmbedder, self).__init__()
        self.embedding = StackedEmbeddings([
            FlairEmbeddings('pubmed-forward', fine_tune=False),
            FlairEmbeddings('pubmed-backward', fine_tune=False),
        ])
