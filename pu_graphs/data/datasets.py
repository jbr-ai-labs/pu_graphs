import typing as ty
import warnings

import dgl
import torch
from torch import LongTensor
from torch.utils.data import Dataset

from . import keys
from .negative_sampling import SamplingStrategy
from .utils import get_edges_as_triplets


class DglGraphDataset(Dataset):

    def __init__(self, graph: dgl.DGLGraph, strategy: ty.Optional[SamplingStrategy] = None):
        self.graph = graph
        self.strategy = strategy

        if self.strategy is not None:
            warnings.warn(
                "Passing sampling strategy directly to dataset class is deprecated, "
                "consider using BatchTransformCallback"
            )

        self.edges = get_edges_as_triplets(self.graph)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, item: int) -> ty.Dict[str, LongTensor]:
        head_idx, tail_idx, relation_idx = self.edges[item]
        data = {
            keys.head_idx: head_idx,
            keys.tail_idx: tail_idx,
            keys.rel_idx: relation_idx
        }
        if self.strategy is None:
            return data

        neg_head_idx, neg_tail_idx = self.strategy.sample(head_idx.item(), tail_idx.item())
        data.update({
            keys.neg_head_idx: torch.tensor(neg_head_idx),
            keys.neg_tail_idx: torch.tensor(neg_tail_idx),
        })
        return data
