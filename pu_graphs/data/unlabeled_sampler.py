from typing import Optional, Mapping, Any

import dgl
import torch

from pu_graphs.data import keys
from pu_graphs.data.negative_sampling import UniformStrategy
from pu_graphs.data.utils import get_edges_as_triplets


class UnlabeledSampler:

    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph
        self.number_of_nodes = self.graph.number_of_nodes()
        self.strategy = UniformStrategy(graph)
        self.edges = get_edges_as_triplets(self.graph)

    def sample_for_batch(self, batch: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """
        Possible sampling strategies:
        1. Batch present: Resample head/tail entity from existing batch
        2. Sample existing triplet, resample head/tail entity
        3. Sample completely random triplet
        :param batch:
        :return:
        """
        new_head_idx, new_tail_idx = self.strategy.sample_batch(
            head_idx=batch[keys.head_idx],
            tail_idx=batch[keys.tail_idx]
        )
        return {
            keys.head_idx: new_head_idx,
            keys.tail_idx: new_tail_idx,
            keys.rel_idx: batch[keys.rel_idx]
        }

    def sample_n_examples(self, n_examples: int) -> Mapping[str, Any]:
        batch = self._sample_batch(n_examples)
        return self.sample_for_batch(batch)

    def _sample_batch(self, n_examples: int) -> Mapping[str, Any]:
        examples = self.edges[torch.randint(0, self.number_of_nodes, [n_examples])]
        head_idx, tail_idx, rel_idx = examples.transpose(0, 1)
        return {
            keys.head_idx: head_idx,
            keys.tail_idx: tail_idx,
            keys.rel_idx: rel_idx
        }
