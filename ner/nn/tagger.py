from logging import getLogger
from typing import Type

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchglyph.dataset import DataLoader
from torchglyph.meter import AverageMeter
from torchglyph.vocab import Vocab
from tqdm import tqdm

from ner.data.dataset import Dataset
from ner.env import setup_env
from ner.monitor import monitor_param_size
from ner.nn.decoder import Decoder, NaiveDecoder
from ner.nn.embedding import Embedding, EmbeddingLayer
from ner.nn.encoder import Encoder, LstmEncoder
from ner.optimizer import Optimizer
from ner.scheduler import Scheduler

logger = getLogger(__name__)

__all__ = [
    'train_tagger',
]


class Tagger(nn.Module):
    def __init__(self, emb_: Embedding = EmbeddingLayer,
                 enc_: Encoder = LstmEncoder,
                 dec_: Decoder = NaiveDecoder, *,
                 num_chunks: int, ctx_dim: int,
                 word_vocab: Vocab, char_vocab: Vocab, tag_vocab: Vocab) -> None:
        super(Tagger, self).__init__()

        self.embedding_layer = emb_(
            word_vocab=word_vocab,
            char_vocab=char_vocab,
            ctx_dim=ctx_dim,
        )
        self.encoder_layer = enc_(
            in_size=self.embedding_layer.embedding_dim,
            num_chunks=num_chunks,
            tag_vocab=tag_vocab,
        )
        self.decoder_layer = dec_(
            num_chunks=num_chunks,
            tag_vocab=tag_vocab,
        )

    def forward(self, batch) -> PackedSequence:
        embedding = self.embedding_layer(batch=batch)
        return self.encoder_layer(embedding)

    def fit(self, batch):
        emissions = self(batch)
        loss = self.decoder_layer.fit(emissions=emissions, batch=batch)
        return loss, {}

    def inference(self, batch):
        emissions = self(batch)
        predictions, selection = self.decoder_layer.decode(emissions=emissions, batch=batch)
        return predictions, selection


def train_tagger(
        setup_env_fn: Type[setup_env],
        data: Dataset,
        model: Type[Tagger],
        optimizer: Optimizer,
        scheduler: Scheduler,
        report_inr: int = -1,
        num_epochs: int = 100,
        grad_norm: float = 5.0,
        **kwargs,
):
    device, out_dir = setup_env_fn(**kwargs['@aku'])

    (train, dev, test), ctx_dim = data(device=device)

    vocabs = train.vocabs
    model = model(
        ctx_dim=ctx_dim,
        num_chunks=train.dataset.num_chunks,
        word_vocab=vocabs.word,
        char_vocab=vocabs.char,
        tag_vocab=vocabs.tag,
    ).to(device)
    monitor_param_size(model=model)

    optimizer = optimizer(model=model)
    logger.info(f'optimizer => {optimizer}')

    scheduler = scheduler(optimizer=optimizer, num_training_steps=len(train) * num_epochs)
    logger.info(f'scheduler => {scheduler}')

    report_inr = max(1, (len(train) + 9) // 10)

    def fit_stage(data_loader: DataLoader, stage_name: str):
        model.train()

        loss_meter = AverageMeter()
        for index, batch in enumerate(tqdm(data_loader, desc=f'{stage_name} {epoch}')):
            loss, _ = model.fit(batch)

            optimizer.zero_grad()
            loss.backward()

            if grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=grad_norm,
                )

            optimizer.step()
            scheduler.step_update()

            loss_meter.update(loss.detach().item())
            if loss_meter.weight is not None and loss_meter.weight >= report_inr:
                logger.info(f'loss => {loss_meter.average:.6f}')
                loss_meter.reset()

        if loss_meter.weight is not None:
            logger.info(f'loss => {loss_meter.average:.6f}')
            loss_meter.reset()

    def inference_stage(data_loader: DataLoader, stage_name: str, index: int):
        model.eval()

        path = out_dir / f'{stage_name}.{index}.txt'
        with path.open('w', encoding='utf-8') as fp:
            for batch in tqdm(data_loader, desc=f'{stage_name} {epoch}'):
                prediction, selection = model.inference(batch)
                data_loader.dataset.dump(fp, batch, prediction, selection)

        _, info = data_loader.dataset.eval(path)

        logger.info(f'{stage_name}.precision => {info["precision"]:.2f}')
        logger.info(f'{stage_name}.recall => {info["recall"]:.2f}')
        logger.info(f'{stage_name}.f1 => {info["f1"]:.2f}')

    for epoch in range(1, 1 + num_epochs):
        fit_stage(train, stage_name='train')

        inference_stage(dev, stage_name='dev', index=epoch)
        inference_stage(test, stage_name='test', index=epoch)

        scheduler.epoch_update()
