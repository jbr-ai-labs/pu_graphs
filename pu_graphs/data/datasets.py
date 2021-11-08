import typing as ty

import torch
from torch import LongTensor
from torch.utils.data import Dataset

import dgl

from .negative_sampling import SamplingStrategy


class DglGraphDataset(Dataset):

    def __init__(self, graph: dgl.DGLGraph, strategy: SamplingStrategy):
        self.graph = graph
        self.strategy = strategy
        self.edges: ty.List[ty.Tuple[LongTensor, LongTensor]] = list(zip(*self.graph.edges()))

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, item: int) -> ty.Dict[str, LongTensor]:
        head_idx, tail_idx = self.edges[item]
        neg_head_idx, neg_tail_idx = self.strategy.sample(head_idx.item(), tail_idx.item())
        return {
            "head_indices": head_idx,
            "tail_indices": tail_idx,
            "neg_head_indices": torch.tensor(neg_head_idx),
            "neg_tail_indices": torch.tensor(neg_tail_idx),
        }
