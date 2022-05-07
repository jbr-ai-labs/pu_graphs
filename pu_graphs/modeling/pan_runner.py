import abc
from typing import Mapping, Any

import torch
from catalyst import dl
from catalyst.utils.torch import any2device
from torch import nn

from pu_graphs.data import keys
from pu_graphs.data.unlabeled_sampler import UnlabeledSampler


class LogitToProbability(nn.Module, abc.ABC):

    def __init__(self, delegate: nn.Module):
        super(LogitToProbability, self).__init__()
        self.delegate = delegate

    @abc.abstractmethod
    def forward_logit(self, *args, **kwargs):
        pass


class SigmoidLogitToProbability(LogitToProbability):

    def forward_logit(self, *args, **kwargs):
        return self.delegate(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.delegate(*args, **kwargs).sigmoid()


class LearnableLogitToProbability(LogitToProbability):

    def __init__(self, delegate):
        super(LearnableLogitToProbability, self).__init__(delegate)
        self.scale = nn.Linear(1, 1)

    def forward_logit(self, *args, **kwargs):
        x = self.delegate(*args, **kwargs)
        return self.scale(x.unsqueeze(-1)).squeeze(-1)

    def forward(self, *args, **kwargs):
        return self.forward_logit(*args, **kwargs).sigmoid()


class PanLoss(nn.Module):

    def __init__(self, alpha: float):
        super(PanLoss, self).__init__()
        self.alpha = alpha

    @staticmethod
    def _cls_term(disc_unlabeled_probs, cls_unlabeled_probs):
        return (
            (1 - cls_unlabeled_probs).log() - cls_unlabeled_probs.log()
        ) * (2 * disc_unlabeled_probs - 1)


class PanDiscriminatorLoss(PanLoss):

    def forward(self, disc_positive_probs, disc_unlabeled_probs, cls_unlabeled_probs):
        disc_term = disc_positive_probs.log() + (1 - disc_unlabeled_probs).log()
        loss = (
            disc_term
            + self.alpha * self._cls_term(
                disc_unlabeled_probs=disc_unlabeled_probs, cls_unlabeled_probs=cls_unlabeled_probs
            )
        ).sum()
        return -loss  # grad ascent


class PanClassifierLoss(PanLoss):

    def forward(self, disc_unlabeled_probs, cls_unlabeled_probs):
        loss = self.alpha * self._cls_term(
            disc_unlabeled_probs=disc_unlabeled_probs, cls_unlabeled_probs=cls_unlabeled_probs
        ).sum()
        return loss

    @staticmethod
    def _cls_term(disc_unlabeled_probs, cls_unlabeled_probs):
        return (
            (1 - cls_unlabeled_probs).log() - cls_unlabeled_probs.log()
        ) * (2 * disc_unlabeled_probs - 1)


def get_pan_loss_by_key(key: str, alpha: float) -> PanLoss:
    if key == PanRunner.DISC_KEY:
        return PanDiscriminatorLoss(alpha)
    if key == PanRunner.CLS_KEY:
        return PanClassifierLoss(alpha)
    raise ValueError(f"Unexpected value for PanLoss key: {key}")


def forward_batch(model, batch):
    return model(
        head_indices=batch[keys.head_idx],
        tail_indices=batch[keys.tail_idx],
        relation_indices=batch[keys.rel_idx]
    )


class PanRunner(dl.Runner):

    DISC_KEY = "disc"
    CLS_KEY = "cls"

    LOSS_DISC_KEY = f"loss_{DISC_KEY}"
    LOSS_CLS_KEY = f"loss_{CLS_KEY}"

    @staticmethod
    def criterion_key(key: str):
        return f"criterion_{key}"

    @staticmethod
    def optimizer_key(key: str):
        return f"optimizer_{key}"

    def __init__(self, unlabeled_sampler: UnlabeledSampler, k: int = 1):
        self.discriminator = None
        self.classifier = None

        # noinspection PyTypeChecker
        self.disc_loss: PanDiscriminatorLoss = None
        # noinspection PyTypeChecker
        self.cls_loss: PanClassifierLoss = None

        self.unlabeled_sampler = unlabeled_sampler
        assert k > 0, "k (classifier update frequency) must be larger than 0"
        self.k = k
        self.step_mod_k = 0
        super(PanRunner, self).__init__()

    def on_experiment_start(self, runner: "IRunner"):
        super(PanRunner, self).on_experiment_start(runner)
        print("Started experiment, initializing discriminator and classifier from model dict")
        self.discriminator = self._model[PanRunner.DISC_KEY]
        self.classifier = self._model[PanRunner.CLS_KEY]

        self.disc_loss = self._criterion[PanRunner.DISC_KEY]
        self.cls_loss = self._criterion[PanRunner.CLS_KEY]

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        self.step_mod_k += 1

        self.discriminator_step(batch)

        if self.step_mod_k == self.k:
            self.step_mod_k = 0
            self.classifier_step()

    def discriminator_step(self, batch):
        # noinspection PyTypeChecker
        unlabeled_data = any2device(
            self.unlabeled_sampler.sample_for_batch(batch), self.device
        )

        disc_positive_probs = forward_batch(self.discriminator, batch)
        disc_unlabeled_probs = forward_batch(self.discriminator, unlabeled_data)
        with torch.no_grad():
            cls_unlabeled_probs = forward_batch(self.classifier, unlabeled_data)

        self.batch.update(
            {
                keys.disc_positive_probs: disc_positive_probs,
                keys.disc_unlabeled_probs: disc_unlabeled_probs,
                keys.cls_unlabeled_probs: cls_unlabeled_probs
            }
        )

        self._optimize(PanRunner.DISC_KEY)

    def classifier_step(self):
        # noinspection PyTypeChecker
        unlabeled_data = any2device(
            self.unlabeled_sampler.sample_n_examples(self.batch_size), self.device
        )

        with torch.no_grad():
            disc_unlabeled_probs = forward_batch(self.discriminator, unlabeled_data)
        cls_unlabeled_probs = forward_batch(self.classifier, unlabeled_data)

        self.batch.update({
            keys.disc_unlabeled_probs: disc_unlabeled_probs,
            keys.cls_unlabeled_probs: cls_unlabeled_probs
        })

        self._optimize(PanRunner.CLS_KEY)

    # noinspection PyUnresolvedReferences
    def _optimize(self, key: str):
        self.callbacks[PanRunner.criterion_key(key)].on_batch_end_manual(self)
        self.callbacks[PanRunner.optimizer_key(key)].on_batch_end_manual(self)
