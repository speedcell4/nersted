import json
from pathlib import Path
from typing import Tuple

__all__ = [
    'Entity', 'SEPARATOR',
    'iter_data',
    'iter_dump',
]

Entity = Tuple[str, int, int]
SEPARATOR = '~~~'


def iter_data(path: Path):
    with path.open(mode='r', encoding='utf-8') as fp:
        while True:
            try:
                words = next(fp).strip().split(' ')

                entities = []
                for entity in next(fp).strip().split('|'):
                    if len(entity) > 0:
                        xy, name = entity.split(' ')
                        x, y = xy.split(',')
                        entities.append((name, int(x), int(y)))
                _ = next(fp)

                yield words, entities

            except StopIteration:
                break


def iter_dump(path: Path):
    with path.open(mode='r', encoding='utf-8') as fp:
        while True:
            try:
                words = next(fp).strip().split(' ')

                selections = []
                while True:
                    raw = next(fp).strip()
                    if raw == SEPARATOR:
                        break
                    selections.append(json.loads(raw))

                targets = []
                raw = next(fp).strip()
                if len(raw) > 0:
                    for item in raw.split('|'):
                        name, x, y = item.split(' ')
                        targets.append((name, int(x), int(y)))

                predictions = []
                while True:
                    raw = next(fp).strip()
                    if raw == SEPARATOR:
                        break
                    prediction = []
                    if len(raw) > 0:
                        for item in raw.split('|'):
                            name, x, y = item.split(' ')
                            prediction.append((name, int(x), int(y)))
                    predictions.append(prediction)

                yield words, selections, targets, predictions

            except StopIteration:
                break
