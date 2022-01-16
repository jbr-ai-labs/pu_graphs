import typing as ty

import dgl
import torch
from torch import LongTensor
from torch.utils.data import Dataset

from .negative_sampling import SamplingStrategy


class DglGraphDataset(Dataset):

    def __init__(self, graph: dgl.DGLGraph, strategy: SamplingStrategy):
        self.graph = graph
        self.strategy = strategy

        head_idx, tail_idx = self.graph.edges()
        relation_idx = self.graph.edata["etype"]

        assert len(head_idx) == len(tail_idx)
        assert len(head_idx) == len(relation_idx)

        self.edges: ty.List[ty.Tuple[LongTensor, LongTensor, LongTensor]] = list(zip(
            *(head_idx, tail_idx, relation_idx)
        ))

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, item: int) -> ty.Dict[str, LongTensor]:
        head_idx, tail_idx, relation_idx = self.edges[item]
        neg_head_idx, neg_tail_idx = self.strategy.sample(head_idx.item(), tail_idx.item())
        return {
            "head_indices": head_idx,
            "tail_indices": tail_idx,
            "relation_indices": relation_idx,
            "neg_head_indices": torch.tensor(neg_head_idx),
            "neg_tail_indices": torch.tensor(neg_tail_idx),
        }
