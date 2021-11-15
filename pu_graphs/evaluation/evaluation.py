import abc
import typing as ty
from typing import Any

import dgl
import scipy
import torch
from catalyst.metrics import IMetric, MRRMetric, AccuracyMetric
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinkPredictionMetric(IMetric, abc.ABC):

    def __init__(self, compute_on_call: bool = True):
        super(LinkPredictionMetric, self).__init__(compute_on_call=compute_on_call)

    @abc.abstractmethod
    def update(self, head_idx, tail_idx, logits, graph) -> Any:
        pass


class LinkPredictionMetricAdapter(LinkPredictionMetric):

    # noinspection PyMissingConstructor
    def __init__(self, metric: IMetric):
        self.metric = metric

    def reset(self) -> None:
        self.metric.reset()

    def update(self, head_idx, tail_idx, logits, graph) -> Any:
        targets = self._transform_targets(targets=tail_idx, graph=graph)
        return self.metric.update(logits=logits, targets=targets)

    def compute(self) -> Any:
        return self.metric.compute()

    def _transform_targets(self, targets, graph):
        return targets

    def __getattr__(self, item):
        return getattr(self.metric, item)


class FilteredLinkPredictionMetric(LinkPredictionMetricAdapter):

    # noinspection PyMissingConstructor
    def __init__(self, metric: LinkPredictionMetric, full_adj_mat: scipy.sparse.csr.csr_matrix):
        self.metric = metric
        self.full_adj_mat = full_adj_mat

    @staticmethod
    def filter_logits(full_adj_mat, logits, head_idx, tail_idx, graph):
        """
        Makes logits for pairs other than (head_idx, tail_idx)
        """
        batch_adj_mat = torch.tensor(full_adj_mat[head_idx].todense())
        targets = torch.nn.functional.one_hot(
            tail_idx,
            num_classes=full_adj_mat.shape[1]
        )
        mask = batch_adj_mat - targets
        return torch.where(
            mask.byte(),
            torch.tensor(float("-inf")).type(logits.dtype),
            logits
        )

    def update(self, head_idx, tail_idx, logits, graph) -> Any:
        logits = FilteredLinkPredictionMetric.filter_logits(
            full_adj_mat=self.full_adj_mat, logits=logits, head_idx=head_idx, tail_idx=tail_idx, graph=graph
        )
        return self.metric.update(logits=logits, head_idx=head_idx, tail_idx=tail_idx, graph=graph)


class MRRLinkPredictionMetric(LinkPredictionMetricAdapter):

    def __init__(self, topk_args=None):
        super(MRRLinkPredictionMetric, self).__init__(metric=MRRMetric(topk_args=topk_args))

    def _transform_targets(self, targets, graph):
        return torch.nn.functional.one_hot(targets, num_classes=graph.number_of_nodes())


class AccuracyLinkPredictionMetric(LinkPredictionMetricAdapter):

    def __init__(self, topk_args):
        super(AccuracyLinkPredictionMetric, self).__init__(metric=AccuracyMetric(topk_args=topk_args))


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
