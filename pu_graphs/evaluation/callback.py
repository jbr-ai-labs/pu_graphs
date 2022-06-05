import typing as ty
from copy import deepcopy

import dgl
import numpy as np
import sparse
import torch
from catalyst import dl
from catalyst.utils.misc import flatten_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu_graphs.evaluation.evaluation import LinkPredictionMetric, FilteredLinkPredictionMetric, MRRLinkPredictionMetric, \
    AccuracyLinkPredictionMetric, AdjustedMeanRankIndex
from pu_graphs.modeling.pan_runner import LogitToProbability


class EvaluationCallback(dl.Callback):
    def __init__(
        self,
        graph: dgl.DGLGraph,
        metrics: ty.Dict[str, LinkPredictionMetric],
        loader: DataLoader,
        loader_key: str,
        is_debug: bool,  # TODO: refactor
        eval_every_epoch: bool = False,
        model_key: ty.Optional[str] = None
    ):
        self.graph = graph
        self.metrics = metrics
        self.loader = loader
        self.loader_key = loader_key
        self.scores = None
        self.is_debug = is_debug
        self.eval_every_epoch = eval_every_epoch
        self.model_key = model_key
        self.was_called = False  # FIXME: some internal catalyst problems
        super(EvaluationCallback, self).__init__(order=dl.CallbackOrder.Metric, node=dl.CallbackNode.Master)

    def _compute_score(self, runner, head_idx, relation_idx):
        model = runner.model if self.model_key is None else runner.model[self.model_key]
        model.eval()

        batch_size = head_idx.shape[0]

        number_of_nodes = self.graph.number_of_nodes()
        all_tail_idx = torch.arange(0, number_of_nodes).to(torch.device(runner.device))

        # shape: [batch_size, 1]
        expanded_head_idx = head_idx.unsqueeze(-1).to(torch.device(runner.device))
        # shape: [batch_size, number_of_nodes]
        expanded_all_tail_idx = all_tail_idx.expand([batch_size, -1])
        # shape: [batch_size, 1]:
        expanded_relation_idx = relation_idx.unsqueeze(-1).to(torch.device(runner.device))

        # shape: [batch_size, number_of_nodes]
        if isinstance(model, LogitToProbability):
            logits = model.forward_logit(
                head_indices=expanded_head_idx,
                tail_indices=expanded_all_tail_idx,
                relation_indices=expanded_relation_idx
            )
        else:
            logits = model.forward(
                head_indices=expanded_head_idx,
                tail_indices=expanded_all_tail_idx,
                relation_indices=expanded_relation_idx
            )

        return logits.to(torch.device("cpu"))

    @torch.no_grad()
    def _compute_metrics(self, runner):
        for batch in tqdm(self.loader, desc="Computing metrics"):
            head_idx, tail_idx, relation_idx = batch["head_indices"], batch["tail_indices"], batch["relation_indices"]

            logits = self._compute_score(runner, head_idx=head_idx, relation_idx=relation_idx)

            for metric in self.metrics.values():
                metric.update(head_idx=head_idx, tail_idx=tail_idx, relation_idx=relation_idx, logits=logits)

        return {
            k: v
            for m in self.metrics.values()
            for k, v in m.compute_key_value().items()
        }

    def on_epoch_end(self, runner: dl.IRunner) -> None:
        if self.loader_key == "test" or not self.eval_every_epoch:
            return

        self._reset_metrics()

        metrics = flatten_dict(self._compute_metrics(runner))
        runner.epoch_metrics[runner.loader_key].update(metrics)
        self._log_metrics(runner, metrics)

    def on_experiment_end(self, runner: dl.IRunner) -> None:
        if (self.loader_key != "test" and self.eval_every_epoch) or self.was_called:
            return
        else:
            self.was_called = True

        self._reset_metrics()

        metrics = flatten_dict(self._compute_metrics(runner))
        self._log_metrics(runner, metrics)

    def _log_metrics(self, runner, metrics):
        kwargs = deepcopy(runner._log_defaults)
        kwargs.update({
            "metrics": metrics,
            "scope": "loader",
            "loader_key": self.loader_key,
        })

        for logger in runner.loggers.values():
            logger.log_metrics(**kwargs)

    def _reset_metrics(self):
        for m in self.metrics.values():
            m.reset()


def evaluation_callback(
    graphs,
    loaders,
    eval_loader_key: str,
    is_debug: bool,
    model_key: ty.Optional[str] = None,
    eval_every_epoch: bool = False
) -> EvaluationCallback:
    full_graph = graphs["train"]
    eval_graph = graphs[eval_loader_key]

    number_of_nodes = full_graph.number_of_nodes()
    number_of_relations = eval_graph.edata["etype"].max().item() + 1

    head_idx, tail_idx = full_graph.edges()
    head_idx = head_idx.numpy()
    tail_idx = tail_idx.numpy()
    relation_idx = full_graph.edata["etype"].numpy()

    coordinates = (relation_idx, head_idx, tail_idx)
    values = np.ones_like(head_idx)

    full_adj_mat = sparse.COO((values, coordinates), shape=[number_of_relations, number_of_nodes, number_of_nodes])

    metrics_builders = {
        "mrr": lambda suf: MRRLinkPredictionMetric(topk_args=[number_of_nodes], suffix=suf),
        "acc": lambda suf: AccuracyLinkPredictionMetric(topk_args=[1, 3, 5, 10, 20], suffix=suf),
        "amri": lambda suf: AdjustedMeanRankIndex(topk_args=[full_graph.number_of_nodes()], suffix=suf)
    }

    metrics = {}
    for k, v in metrics_builders.items():
        metrics[k] = v("")
        metrics[f"{k}_filtered"] = FilteredLinkPredictionMetric(metric=v("_filtered"), full_adj_mat=full_adj_mat)

    return EvaluationCallback(
        graph=eval_graph,
        metrics=metrics,
        loader=loaders[eval_loader_key],
        loader_key=eval_loader_key,
        is_debug=is_debug,
        model_key=model_key,
        eval_every_epoch=eval_every_epoch
    )
