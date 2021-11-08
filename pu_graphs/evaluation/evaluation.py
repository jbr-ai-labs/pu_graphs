import dgl
import torch
from catalyst import dl
from catalyst.metrics import reciprocal_rank
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def compute_mrr_for_loader(graph: dgl.DGLGraph, model, loader: DataLoader):
    sum_rr = 0
    number_of_nodes = graph.number_of_nodes()
    all_tail_idx = torch.arange(0, number_of_nodes)
    for batch in tqdm(loader):
        head_idx, tail_idx = batch["head_indices"], batch["tail_indices"]

        expanded_head_idx = head_idx.unsqueeze(-1).expand([-1, len(all_tail_idx)])

        batch_size = len(head_idx)
        expanded_all_tail_idx = all_tail_idx.expand([batch_size, -1])

        scores = model(head_indices=expanded_head_idx, tail_indices=expanded_all_tail_idx)["scores"]
        reciprocal_ranks = reciprocal_rank(
            outputs=scores,
            targets=torch.nn.functional.one_hot(tail_idx, num_classes=number_of_nodes),
            k=len(all_tail_idx)  # TODO: maybe move k to parameters
        )

        sum_rr += reciprocal_ranks.sum()

    return sum_rr / len(loader)


class LoaderMrrCallback(dl.Callback):

    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph
        super().__init__(order=dl.CallbackOrder.Metric, node=dl.CallbackNode.Master)

    def on_loader_end(self, runner: "IRunner") -> None:
        runner.loader_metrics["mrr"] = compute_mrr_for_loader(
            graph=self.graph, model=runner.model, loader=runner.loader
        )
