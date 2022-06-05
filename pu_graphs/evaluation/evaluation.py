import abc
import typing as ty
from typing import Any, Dict

import dgl
import scipy
import torch
from catalyst.metrics import IMetric, MRRMetric, AccuracyMetric, ICallbackBatchMetric, \
    process_recsys_components, FunctionalBatchMetric
from torch.utils.data import DataLoader
from tqdm import tqdm


class LinkPredictionMetric(IMetric, abc.ABC):

    @abc.abstractmethod
    def update(
        self,
        head_idx: torch.LongTensor,
        tail_idx: torch.LongTensor,
        relation_idx: torch.LongTensor,
        logits: torch.Tensor
    ) -> Any:
        """

        :param head_idx: shape [batch_size]
        :param tail_idx: shape [batch_size]
        :param relation_idx: shape [batch_size]
        :param logits: shape [batch_size, number_of_nodes]
        :return:
        """
        pass


class LinkPredictionMetricAdapter(LinkPredictionMetric):

    # noinspection PyMissingConstructor
    def __init__(self, metric: IMetric):
        self.metric = metric

    def reset(self) -> None:
        self.metric.reset()

    def update(self, head_idx, tail_idx, relation_idx, logits) -> Any:
        targets = self._transform_targets(
            head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, logits=logits
        )
        return self.metric.update(logits=logits, targets=targets)

    def compute(self) -> Any:
        return self.metric.compute()

    def _transform_targets(self, head_idx, tail_idx, relation_idx, logits):
        return tail_idx

    def __getattr__(self, item):
        return getattr(self.metric, item)


class FilteredLinkPredictionMetric(LinkPredictionMetricAdapter):

    # noinspection PyMissingConstructor
    def __init__(self, metric: LinkPredictionMetric, full_adj_mat: scipy.sparse.csr.csr_matrix):
        self.metric = metric
        self.full_adj_mat = full_adj_mat

    @staticmethod
    def filter_logits(full_adj_mat, logits, head_idx, tail_idx, relation_idx) -> torch.Tensor:
        """
        Makes logits for pairs other than (head_idx, tail_idx)
        """
        batch_adj_mat = torch.tensor(full_adj_mat[relation_idx, head_idx].todense())
        targets = torch.nn.functional.one_hot(
            tail_idx,
            num_classes=full_adj_mat.shape[1]
        )

        # target may be absent from full_adj_mat since we take train as full graph
        mask = (batch_adj_mat - targets).maximum(torch.tensor(0))

        return torch.where(
            mask.byte(),
            torch.tensor(float("-inf")).type(logits.dtype),
            logits
        )

    def update(self, head_idx, tail_idx, relation_idx, logits) -> Any:
        logits = FilteredLinkPredictionMetric.filter_logits(
            full_adj_mat=self.full_adj_mat,
            logits=logits,
            head_idx=head_idx,
            tail_idx=tail_idx,
            relation_idx=relation_idx
        )
        return self.metric.update(logits=logits, head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx)


class MRRLinkPredictionMetric(LinkPredictionMetricAdapter):

    def __init__(self, topk_args=None, suffix: str = ""):
        super(MRRLinkPredictionMetric, self).__init__(metric=MRRMetric(topk_args=topk_args, suffix=suffix))

    def _transform_targets(self, head_idx, tail_idx, relation_idx, logits):
        number_of_nodes = logits.shape[-1]
        return torch.nn.functional.one_hot(tail_idx, num_classes=number_of_nodes)

    def compute_key_value(self):
        result = self.metric.compute_key_value()
        metric = self.metric
        key = f"{metric.prefix}{metric.metric_name}{metric.suffix}"
        if key not in result:
            max_k = max(metric.topk_args)
            max_k_key = f"{metric.prefix}{metric.metric_name}{max_k:02d}{metric.suffix}"
            result[key] = result[max_k_key]
        return result


class AccuracyLinkPredictionMetric(LinkPredictionMetricAdapter):

    def __init__(self, topk_args, suffix: str = ""):
        super(AccuracyLinkPredictionMetric, self).__init__(metric=AccuracyMetric(topk_args=topk_args, suffix=suffix))


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
            metric.update(head_idx=head_idx, tail_idx=tail_idx, logits=logits)

    return {
        k: {"mean": mean, "std": std}
        for k, (mean, std) in map(
            lambda item: (item[0], item[1].compute()),
            metrics.items()
        )
    }


@torch.no_grad()
def compute_score_based_metrics_for_loader_optimized(
    graph: dgl.DGLGraph,
    model,
    loader: DataLoader,
    metrics: ty.Dict[str, LinkPredictionMetric]
):
    number_of_nodes = graph.number_of_nodes()
    all_tail_idx = torch.arange(0, number_of_nodes)
    scores = torch.empty(number_of_nodes, number_of_nodes)
    batch_size = loader.batch_size
    for i in tqdm(range(0, number_of_nodes, batch_size)):
        head_idx = torch.arange(i, i + batch_size)
        expanded_head_idx = head_idx.unsqueeze(-1).expand([-1, len(all_tail_idx)])

        actual_batch_size = head_idx.shape[0]
        expanded_all_tail_idx = all_tail_idx.expand([actual_batch_size, -1])

        logits = model(head_indices=expanded_head_idx, tail_indices=expanded_all_tail_idx)
        scores[i:i+batch_size] = logits

    for batch in tqdm(loader):
        head_idx, tail_idx = batch["head_indices"], batch["tail_indices"]

        logits = scores[head_idx, tail_idx]

        for metric in metrics.values():
            metric.update(head_idx=head_idx, tail_idx=tail_idx, logits=logits)

    return {
        k: {"mean": mean, "std": std}
        for k, (mean, std) in map(
            lambda item: (item[0], item[1].compute()),
            metrics.items()
        )
    }


# Not used currently but may be useful later
def optimistic_rank(logits: torch.Tensor, target_idx: torch.Tensor, k: int):
    mask = logits > logits[target_idx]
    ranks = mask.sum(-1) + 1
    ranks = ranks.where(
        ranks < k + 1,  # ranks - 1 < k, since we want not more than k elements before us
        ranks, torch.tensor(0).type(ranks.dtype)
    )
    return ranks


def pessimistic_rank(logits: torch.Tensor, target_idx: torch.Tensor, k: int):
    mask = logits >= logits[target_idx]
    ranks = mask.sum(-1)
    ranks = ranks.where(ranks <= k, ranks, torch.tensor(0).type(ranks.dtype))
    return ranks


def realistic_rank(logits: torch.Tensor, target_idx: torch.Tensor, k: int):
    return (optimistic_rank(logits, target_idx, k) + pessimistic_rank(logits, target_idx, k)) / 2


# this one is used by catalyst metrics
def non_deterministic_rank(logits: torch.Tensor, target_idx: torch.Tensor, k: int):
    """
    Rank is 0 when it is larger than k
    """
    targets = torch.nn.functional.one_hot(target_idx, num_classes=logits.shape[-1])
    sorted_targets = process_recsys_components(outputs=logits, targets=targets)[:, :k]

    values, indices = torch.max(sorted_targets, dim=1)
    indices += 1
    indices[values == 0] = 0

    return indices


class AdjustedMeanRankIndex(LinkPredictionMetric, ICallbackBatchMetric):

    def __init__(
        self,
        topk_args: ty.List[int] = None,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
        rank_fn: ty.Optional[ty.Callable] = None
    ):
        ICallbackBatchMetric.__init__(self, compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name_mean = f"{self.prefix}amri{self.suffix}"
        self.metric_name_std = f"{self.prefix}amri{self.suffix}/std"
        self.topk_args: ty.List[int] = sorted(topk_args or [1])
        self.rank_fn = rank_fn if rank_fn is not None else non_deterministic_rank
        self.metrics = [
             FunctionalBatchMetric(
                 AdjustedMeanRankIndex.get_amri_at_k_fn(self.rank_fn, k),
                 metric_key=f"amri_{k}",
                 prefix=prefix,
                 suffix=suffix
             ) for k in self.topk_args
        ]

    @staticmethod
    def get_amri_at_k_fn(rank_fn: ty.Callable, k: int):
        def metric_fn(tail_idx, logits):
            ranks = rank_fn(logits=logits, target_idx=tail_idx, k=k)
            n_tails = logits.shape[-1]
            ranks[ranks == 0] = n_tails
            value = 1 - (2 / (n_tails - 1)) * ranks.sub(1).float().mean()
            return value
        return metric_fn

    def update(self, head_idx, tail_idx, relation_idx, logits) -> Any:
        return [
            m.update(
                batch_size=logits.shape[0],
                tail_idx=tail_idx,
                logits=logits
            )
            for m in self.metrics
        ]

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

    def compute(self) -> Any:
        means, stds = zip(*(metric.compute() for metric in self.metrics))
        return means, stds

    def update_key_value(self, head_idx, tail_idx, logits) -> Dict[str, float]:
        values = self.update(head_idx, tail_idx, logits)
        output = {
            m.metric_name: v
            for m, v in zip(self.metrics, values)
        }
        output[self.metric_name_mean] = output[self.metrics[-1].metric_name]
        return output

    def compute_key_value(self) -> Dict[str, float]:
        values = {}
        for m in self.metrics:
            values.update(m.compute_key_value())

        values[self.metric_name_mean] = values[self.metrics[-1].metric_name]
        values[self.metric_name_std] = values[f"{self.metrics[-1].metric_name}/std"]

        return values
