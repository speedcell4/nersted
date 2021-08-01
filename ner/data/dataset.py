import itertools
import json
from abc import ABCMeta
from logging import getLogger
from pathlib import Path
from typing import Iterable, Any, Type, Union

import torch
from aku import Literal
from torch.nn.utils.rnn import PackedSequence
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.pipe import RawPipe
from torchglyph.proc import LoadVectors
from tqdm import tqdm

from ner import project_data_dir
from ner.data.ctx import BERT_DIM, FLAIR_DIM
from ner.data.ctx import Embedder, BertEmbedder, FlairNewsEmbedder, BioBertEmbedder, FlairPubmedEmbedder
from ner.data.pipe import WordPipe, CharPipe, TagPipe, CtxPipe
from ner.data.proc import LoadPubMed, LoadGlove
from ner.data.scheme import entities_to_innermost_first_tags, entities_to_outermost_first_tags
from ner.data.utils import iter_data, SEPARATOR, iter_dump
from ner.meter import ClassificationMeter

logger = getLogger(__name__)

__all__ = [
    'Dataset',
    'ace2004', 'ace2005', 'genia',
    'ACE2004', 'ACE2005', 'GENIA',
]


class NestedNER(Dataset, metaclass=ABCMeta):
    num_chunks: int
    bert_embeder: Type[Embedder]
    flair_embeder: Type[Embedder]

    @classmethod
    def load_word_vectors(cls) -> LoadVectors:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path, scheme: str, use_bert: bool, use_flair: bool, **kwargs) -> Iterable[Any]:
        scheme = {
            'outer': entities_to_outermost_first_tags,
            'inner': entities_to_innermost_first_tags,
        }[scheme]

        for (words, entities), bert, flair in zip(
                iter_data(path=path),
                cls.cache_bert(path=path, batch_size=32, use_bert=use_bert),
                cls.cache_flair(path=path, batch_size=32, use_flair=use_flair),
        ):
            tags = scheme(entities=entities, length=len(words), num_chunks=cls.num_chunks)

            if use_flair:
                yield [words, entities, tags, torch.cat([bert, flair], dim=-1)]
            elif use_bert:
                yield [words, entities, tags, bert]
            else:
                yield [words, entities, tags]

    @classmethod
    def cache_ctx(cls, path: Path, batch_size: int, use_ctx: bool, name: str, embeder):
        if not use_ctx:
            return itertools.repeat(None)

        cache_path = path.with_name(f'{path.name}.{name}')

        if cache_path.exists():
            logger.info(f'loading from {cache_path}')
            return torch.load(cache_path)
        else:
            ctx_embeder = embeder()

            words, ctx_embedding = [], []
            for tokens, _ in tqdm(iter_data(path=path), desc=f'caching {name} {path}'):
                words.append(tokens)
                if len(words) > batch_size:
                    ctx_embedding.extend(ctx_embeder(words))
                    words = []

            if len(words) > 0:
                ctx_embedding.extend(ctx_embeder(words))

            logger.info(f'saving {len(ctx_embedding)} to {cache_path}')
            torch.save(ctx_embedding, cache_path)
            return ctx_embedding

    @classmethod
    def cache_bert(cls, path: Path, batch_size: int, use_bert: bool):
        return cls.cache_ctx(
            path=path, batch_size=batch_size,
            use_ctx=use_bert, name='bert', embeder=cls.bert_embeder,
        )

    @classmethod
    def cache_flair(cls, path: Path, batch_size: int, use_flair: bool):
        return cls.cache_ctx(
            path=path, batch_size=batch_size,
            use_ctx=use_flair, name='flair', embeder=cls.flair_embeder,
        )

    def dump(self, fp, batch, predictions: PackedSequence, selections: PackedSequence, **kwargs) -> None:
        predictions = self.pipes['tag'].inv(predictions)
        selections = self.pipes['tag'].inv_num(selections)

        for ws, t, ps, ss in zip(batch.raw_word, batch.raw_entity, predictions, selections):
            print(' '.join([f'{token}_{index}' for index, token in enumerate(ws)]), file=fp)
            for s in ss:
                print(json.dumps(s), file=fp)
            print(SEPARATOR, file=fp)
            print('|'.join([f'{name} {x} {y - 1}' for name, x, y in t]), file=fp)
            for p in ps:
                print('|'.join([f'{name} {x} {y}' for name, x, y in p]), file=fp)
            print(SEPARATOR, file=fp)

    def eval(self, path: Path, **kwargs):
        meter = ClassificationMeter()
        for _, _, targets, predictions in iter_dump(path):
            targets = set(targets)
            predictions = set(p for prediction in predictions for p in prediction)

            meter.update(
                value=len(targets & predictions),
                target_weight=len(targets),
                prediction_weight=len(predictions),
            )

        return meter.f1, {
            'precision': meter.precision,
            'recall': meter.recall,
            'f1': meter.f1,
        }

    @classmethod
    def paths(cls, root: Path = project_data_dir, **kwargs):
        name = cls.__name__.lower()
        train = root / name / f'{name}.train'
        dev = root / name / f'{name}.dev'
        test = root / name / f'{name}.test'
        return train, dev, test

    @classmethod
    def new(cls, batch_size: int = 32, use_bert: bool = False, use_flair: bool = False,
            scheme: Literal['inner', 'outer'] = 'inner', *, device: torch.device):

        word = WordPipe(device=device).with_(
            vocab=... + cls.load_word_vectors(),
        )
        char = CharPipe(device=device)
        tag = TagPipe(device=device)

        pipes = [
            dict(word=word, char=char, raw_word=RawPipe()),
            dict(raw_entity=RawPipe()),
            dict(tag=tag, raw_tag=RawPipe()),
        ]

        if use_bert:
            pipes.append(dict(ctx=CtxPipe(device=device)))

        for pipe in pipes:
            for key, value in pipe.items():
                logger.info(f'{key} => {value}')

        train, dev, test = cls.paths()
        train = cls(path=train, pipes=pipes, scheme=scheme, use_bert=use_bert, use_flair=use_flair)
        dev = cls(path=dev, pipes=pipes, scheme=scheme, use_bert=use_bert, use_flair=use_flair)
        test = cls(path=test, pipes=pipes, scheme=scheme, use_bert=use_bert, use_flair=use_flair)

        word.build_vocab(train, dev, test, name='word')
        char.build_vocab(train, name='char')
        tag.build_vocab(train, dev, test, name='tag')

        bert_dim = BERT_DIM if use_bert else 0
        flair_dim = FLAIR_DIM if use_flair else 0
        ctx_dim = bert_dim + flair_dim

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size,
            shuffle=True, drop_last=False,
        ), ctx_dim


class ACE2004(NestedNER):
    num_chunks = 6
    bert_embeder = BertEmbedder
    flair_embeder = FlairNewsEmbedder

    @classmethod
    def load_word_vectors(cls):
        return LoadGlove(str.lower, name='6B', dim=100)


def ace2004(fn: Type[ACE2004.new], *args, **kwargs):
    return fn(*args, **kwargs)


class ACE2005(NestedNER):
    num_chunks = 6
    bert_embeder = BertEmbedder
    flair_embeder = FlairNewsEmbedder

    @classmethod
    def load_word_vectors(cls):
        return LoadGlove(str.lower, name='6B', dim=100)


def ace2005(fn: Type[ACE2005.new], *args, **kwargs):
    return fn(*args, **kwargs)


class GENIA(NestedNER):
    num_chunks = 4
    bert_embeder = BioBertEmbedder
    flair_embeder = FlairPubmedEmbedder

    @classmethod
    def load_word_vectors(cls):
        return LoadPubMed(str.lower)


def genia(fn: Type[GENIA.new], *args, **kwargs):
    return fn(*args, **kwargs)


Dataset = Union[
    Type[ace2004],
    Type[ace2005],
    Type[genia],
]
