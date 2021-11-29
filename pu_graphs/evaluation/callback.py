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
        is_debug: bool  # TODO: refactor
    ):
        self.graph = graph
        self.metrics = metrics
        self.loader = loader
        self.loader_key = loader_key
        self.scores = None
        self.is_debug = is_debug
        self.was_called = False  # FIXME: some internal catalyst problems
        super(EvaluationCallback, self).__init__(order=dl.CallbackOrder.ExternalExtra, node=dl.CallbackNode.Master)

    @torch.no_grad()
    def _compute_scores(self, runner: dl.IRunner):
        model = runner.model
        model.eval()

        number_of_nodes = self.graph.number_of_nodes()
        all_tail_idx = torch.arange(0, number_of_nodes).to(torch.device(runner.device))
        self.scores = torch.empty(number_of_nodes, number_of_nodes)
        batch_size = self.loader.batch_size

        for i in tqdm(range(0, number_of_nodes, batch_size), desc="Computing scores"):
            to = min(i + batch_size, number_of_nodes)
            head_idx = torch.arange(i, to)
            expanded_head_idx = head_idx.unsqueeze(-1).expand([-1, len(all_tail_idx)]).to(torch.device(runner.device))

            actual_batch_size = head_idx.shape[0]
            expanded_all_tail_idx = all_tail_idx.expand([actual_batch_size, -1])

            logits = model(head_indices=expanded_head_idx, tail_indices=expanded_all_tail_idx)
            self.scores[i:to] = logits

    @torch.no_grad()
    def _compute_metrics(self):
        for batch in tqdm(self.loader, desc="Computing metrics"):
            head_idx, tail_idx = batch["head_indices"], batch["tail_indices"]

            logits = self.scores[head_idx]

            for metric in self.metrics.values():
                metric.update(head_idx=head_idx, tail_idx=tail_idx, logits=logits)

        return {
            k: v
            for m in self.metrics.values()
            for k, v in m.compute_key_value().items()
        }

    def on_experiment_end(self, runner: dl.IRunner) -> None:
        if self.was_called:
            return
        else:
            self.was_called = True

        self._compute_scores(runner)
        print("SCORES COMPUTED")
        metrics = flatten_dict(self._compute_metrics())
        print("METRICS: ", metrics)

        kwargs = deepcopy(runner._log_defaults)
        kwargs.update({
            "metrics": metrics,
            "scope": "loader",
            "loader_key": self.loader_key,
        })

        for logger in runner.loggers.values():
            logger.log_metrics(**kwargs)
