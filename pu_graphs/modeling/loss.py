import torch
from torch import nn, Tensor


class PositiveNegativeLoss(nn.Module):

    def __init__(self, surrogate_loss, pi: float):
        super().__init__()
        self.surrogate_loss = surrogate_loss
        self.pi = pi

    def forward(self, logits, labels):
        positive_logits = logits[labels == 1]
        negative_logits = logits[labels == 0]

        positive_loss = self.pi * self.surrogate_loss(positive_logits).mean()
        negative_loss = (1 - self.pi) * self.surrogate_loss(-negative_logits).mean()

        loss = positive_loss + negative_loss

        return loss


class UnbiasedPULoss(nn.Module):
    """
    https://arxiv.org/pdf/1901.10155.pdf
    """
    def __init__(self, surrogate_loss, pi: float, is_non_negative: bool = False):
        super().__init__()
        self.surrogate_loss = surrogate_loss
        self.pi = pi
        self.is_non_negative = is_non_negative

    def forward(self, logits, labels):
        positive_logits = logits[labels == 1]
        unlabeled_logits = logits[labels == 0]

        positive_loss = self.pi * self.surrogate_loss(positive_logits).mean()
        negative_loss = (
            self.surrogate_loss(-unlabeled_logits).mean()
            - self.pi * self.surrogate_loss(-positive_logits).mean()
        )
        loss = positive_loss + (
            negative_loss.maximum(torch.tensor(0))
            if self.is_non_negative
            else negative_loss
        )

        return loss


def sigmoid_loss(t: Tensor):
    return 1 / (1 + t.exp())


def logistic_loss(t: Tensor):
    return t.neg().exp().log1p()


class MarginBasedLoss(nn.Module):

    def __init__(self, margin: float):
        super(MarginBasedLoss, self).__init__()
        self.margin = margin

    def forward(self, logits, labels):
        positive_logits = logits[labels == 1]  # Shape [batch_size]
        negative_logits = logits[labels == 0]  # Shape [batch_size]

        score_distance = negative_logits.unsqueeze(-1) - positive_logits  # Shape [batch_size, batch_size]
        score_distance.add_(self.margin)

        loss = score_distance.maximum(torch.tensor(0)).sum()

        return loss


class PanDiscriminatorLoss(nn.Module):
    def forward(self, logits_disc, logits_cls, labels):
        pass
