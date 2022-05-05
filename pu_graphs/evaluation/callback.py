import typing as ty
from copy import deepcopy

import dgl
import torch
from catalyst import dl
from catalyst.utils.misc import flatten_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from pu_graphs.evaluation.evaluation import LinkPredictionMetric


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
        super(EvaluationCallback, self).__init__(order=dl.CallbackOrder.ExternalExtra, node=dl.CallbackNode.Master)

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
        logits = model(
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

