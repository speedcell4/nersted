from typing import List, Iterator

from ner.data.utils import Entity

__all__ = [
    'entity_to_tags',
    'outermost_first_entities_iter', 'entities_to_outermost_first_tags',
    'innermost_first_entities_iter', 'entities_to_innermost_first_tags',
]


def entity_to_tags(entities: List[Entity], length: int) -> List[str]:
    tags = ['O'] * length

    for name, x, y in entities:
        if x + 1 == y:
            tags[x] = f'S-{name}'
        else:
            tags[x] = f'B-{name}'
            for index in range(x + 1, y - 1):
                tags[index] = f'I-{name}'
            tags[y - 1] = f'E-{name}'

    return tags


def outermost_first_entities_iter(entities: List[Entity], num_chunks: int) -> Iterator[List[Entity]]:
    entities = sorted(list(set(entities)), key=lambda item: (item[1], -item[2], item[0]))

    orders, children, stack = [0] * len(entities), [[] for _ in entities], []
    for index, (_, x, _) in enumerate(entities):
        while len(stack) > 0 and entities[stack[-1]][2] <= x:
            del stack[-1]
        if len(stack) > 0:
            orders[index] += 1
            children[stack[-1]].append(index)
        stack.append(index)

    ans = []
    for _ in range(num_chunks):
        for index, (order, entity) in enumerate(zip(orders[::], entities)):
            if order == 0:
                ans.append(entity)
                orders[index] += 1
                for child in children[index]:
                    orders[child] -= 1
        yield ans
        ans = []


def entities_to_outermost_first_tags(entities: List[Entity], length: int, num_chunks: int) -> List[List[str]]:
    return [
        entity_to_tags(entity, length=length)
        for entity in outermost_first_entities_iter(entities=entities, num_chunks=num_chunks)
    ]


def innermost_first_entities_iter(entities: List[Entity], num_chunks: int) -> Iterator[List[Entity]]:
    entities = sorted(list(set(entities)), key=lambda item: (item[1], -item[2], item[0]))

    orders, parent, stack = [0] * len(entities), [[] for _ in entities], []
    for index, (_, x, _) in enumerate(entities):
        while len(stack) > 0 and entities[stack[-1]][2] <= x:
            del stack[-1]
        if len(stack) > 0:
            orders[stack[-1]] += 1
            parent[index].append(stack[-1])
        stack.append(index)

    ans = []
    for _ in range(num_chunks):
        for index, (order, entity) in enumerate(zip(orders[::], entities)):
            if order == 0:
                ans.append(entity)
                orders[index] += 1
                for child in parent[index]:
                    orders[child] -= 1
        yield ans
        ans = []


def entities_to_innermost_first_tags(entities: List[Entity], length: int, num_chunks: int) -> List[List[str]]:
    return [
        entity_to_tags(entity, length=length)
        for entity in innermost_first_entities_iter(entities=entities, num_chunks=num_chunks)
    ]
