import abc
import typing as ty
from typing import Any

import dgl
import torch
from catalyst.metrics import IMetric, MRRMetric, AccuracyMetric
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinkPredictionMetric(IMetric, abc.ABC):

    def __init__(self, compute_on_call: bool = True):
        super(LinkPredictionMetric, self).__init__(compute_on_call=compute_on_call)


class LinkPredictionMetricAdapter(LinkPredictionMetric):

    def __init__(self, metric: IMetric, compute_on_call: bool = True):
        super(LinkPredictionMetricAdapter, self).__init__(compute_on_call=compute_on_call)
        self.metric = metric

    def reset(self) -> None:
        self.metric.reset()

    def update(self, logits, targets, graph) -> Any:
        targets = self._get_targets(targets=targets, graph=graph)
        return self.metric.update(logits=logits, targets=targets)

    def compute(self) -> Any:
        return self.metric.compute()

    @abc.abstractmethod
    def _get_targets(self, targets, graph):
        pass


class MRRLinkPredictionMetric(LinkPredictionMetricAdapter):

    def __init__(self, topk_args):
        super(MRRLinkPredictionMetric, self).__init__(metric=MRRMetric(topk_args=topk_args))

    def _get_targets(self, targets, graph):
        return torch.nn.functional.one_hot(targets, num_classes=graph.number_of_nodes())


class AccuracyLinPredictionMetric(LinkPredictionMetricAdapter):

    def __init__(self, topk_args):
        super(AccuracyLinPredictionMetric, self).__init__(metric=AccuracyMetric(topk_args=topk_args))

    def _get_targets(self, targets, graph):
        return targets


@torch.no_grad()
def compute_score_based_metrics_for_loader(
    graph: dgl.DGLGraph,
    model,
    loader: DataLoader,
    metrics: ty.Dict[str, LinkPredictionMetric]
):
    number_of_nodes = graph.number_of_nodes()
    all_tail_idx = torch.arange(0, number_of_nodes)
    for batch in tqdm(loader):
        head_idx, tail_idx = batch["head_indices"], batch["tail_indices"]

        expanded_head_idx = head_idx.unsqueeze(-1).expand([-1, len(all_tail_idx)])

        batch_size = len(head_idx)
        expanded_all_tail_idx = all_tail_idx.expand([batch_size, -1])

        logits = model(head_indices=expanded_head_idx, tail_indices=expanded_all_tail_idx)

        for metric in metrics.values():
            metric.update(logits=logits, targets=tail_idx, graph=graph)

    return {
        k: {"mean": mean, "std": std}
        for k, (mean, std) in map(
            lambda item: (item[0], item[1].compute()),
            metrics.items()
        )
    }
