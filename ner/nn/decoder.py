from abc import ABCMeta
from typing import Type, Union, Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence
from torchglyph.vocab import Vocab
from torchlatent.functional import logsumexp

from ner.nn.crf import CrfDecoder, CRF

__all__ = [
    'Decoder',
    'NaiveDecoder',
    'MaxDecoder',
    'LogSumExpDecoder',
]


class DecoderABC(nn.Module, metaclass=ABCMeta):
    def __init__(self, crf: CRF = CrfDecoder, *, num_chunks: int, tag_vocab: Vocab) -> None:
        super(DecoderABC, self).__init__()

        self.num_chunks = num_chunks
        self.crf = crf(tag_vocab=tag_vocab)

    def fit(self, emissions: PackedSequence, batch) -> PackedSequence:
        raise NotImplementedError

    def decode(self, emissions: PackedSequence, batch) -> Tuple[PackedSequence, PackedSequence]:
        raise NotImplementedError


class NaiveDecoder(DecoderABC):
    def fit(self, emissions: PackedSequence, batch) -> Tensor:
        log_prob = self.crf.fit(emissions=emissions, tags=batch.tag)
        return log_prob.sum().neg()

    def decode(self, emissions: PackedSequence, batch) -> Tuple[PackedSequence, PackedSequence]:
        selected_index = torch.arange(self.num_chunks, dtype=torch.long, device=emissions.data.device)
        selected_index = selected_index[None, :].expand((emissions.data.size()[0], -1))
        selections = emissions._replace(data=selected_index)

        predictions = self.crf.decode(emissions=emissions)

        return predictions, selections


class MaxDecoder(DecoderABC):
    @staticmethod
    def select_chunk(emissions: Tensor, padding_mask: Tensor) -> Tuple[Tensor, Tensor]:
        emissions, padding_mask = torch.broadcast_tensors(emissions, padding_mask)  # [b, c, n]

        masked_emissions = torch.masked_fill(emissions, mask=padding_mask, value=-float('inf'))
        selected_emission, selected_index = masked_emissions.max(dim=-2, keepdim=True)
        return selected_emission, selected_index

    @staticmethod
    def update_mask(padding_mask: Tensor, selected_index: Tensor, tag: Tensor) -> Tuple[Tensor, Tensor]:
        index = torch.gather(selected_index, dim=-1, index=tag[:, None, None])
        return torch.scatter(padding_mask, dim=-2, index=index, value=True), index[:, 0, 0]

    def fit(self, emissions: PackedSequence, batch) -> PackedSequence:
        padding_mask = torch.zeros(
            (emissions.data.size()[0], emissions.data.size()[1], 1),
            dtype=torch.bool, device=emissions.data.device, requires_grad=False,
        )

        selected_emissions = []
        for index in range(self.num_chunks):
            selected_emission, selected_index = self.select_chunk(emissions.data, padding_mask)
            padding_mask, _ = self.update_mask(padding_mask, selected_index, batch.tag.data[:, index])

            selected_emissions.append(selected_emission)

        selected_emissions = emissions._replace(data=torch.cat(selected_emissions, dim=-2))

        log_prob = self.crf.fit(emissions=selected_emissions, tags=batch.tag)
        return log_prob.sum().neg()

    def decode(self, emissions: PackedSequence, batch) -> Tuple[PackedSequence, PackedSequence]:
        padding_mask = torch.zeros(
            (emissions.data.size()[0], emissions.data.size()[1], 1),
            dtype=torch.bool, device=emissions.data.device, requires_grad=False,
        )

        predictions, selected_indices = [], []
        for index in range(self.num_chunks):
            selected_emission, selected_index = self.select_chunk(emissions.data, padding_mask)
            prediction = self.crf.decode(emissions=emissions._replace(data=selected_emission))
            padding_mask, selected_index = self.update_mask(padding_mask, selected_index, prediction.data[:, 0])

            predictions.append(prediction.data)
            selected_indices.append(selected_index)

        predictions = emissions._replace(data=torch.cat(predictions, dim=-1))
        selections = emissions._replace(data=torch.stack(selected_indices, dim=-1))
        return predictions, selections


class LogSumExpDecoder(MaxDecoder):
    @staticmethod
    def select_chunk(emissions: Tensor, padding_mask: Tensor) -> Tuple[Tensor, Tensor]:
        emissions, padding_mask = torch.broadcast_tensors(emissions, padding_mask)  # [b, c, n]

        masked_emissions = torch.masked_fill(emissions, mask=padding_mask, value=-float('inf'))
        selected_index = masked_emissions.argmax(dim=-2, keepdim=True)
        selected_emission = logsumexp(masked_emissions, dim=-2, keepdim=True)
        return selected_emission, selected_index


Decoder = Union[
    Type[NaiveDecoder],
    Type[MaxDecoder],
    Type[LogSumExpDecoder],
]
