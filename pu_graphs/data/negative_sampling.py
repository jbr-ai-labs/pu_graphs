import abc
import random
import typing as ty

import dgl
import torch


class SamplingStrategy(abc.ABC):

    @abc.abstractmethod
    def sample(self, src_idx, dst_idx):
        pass


def sample_not_equal_to(original, sampler):
    sampled = sampler()
    while sampled == original:
        sampled = sampler()
    return sampled


class UniformStrategy(SamplingStrategy):

    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph
        self.number_of_nodes = self.graph.number_of_nodes()

    def _sampler(self):
        return random.randint(0, self.number_of_nodes - 1)

    def sample(self, src_idx: int, dst_idx: int) -> ty.Tuple[int, int]:
        if random.random() < 0.5:
            return src_idx, sample_not_equal_to(src_idx, self._sampler)
        else:
            return sample_not_equal_to(dst_idx, self._sampler), dst_idx

    def sample_batch(
        self,
        head_idx: torch.LongTensor,
        tail_idx: torch.LongTensor
    ) -> ty.Tuple[torch.LongTensor, torch.LongTensor]:
        mask = torch.randint_like(head_idx, low=0, high=2).bool()
        samples = torch.randint_like(head_idx, low=0, high=self.number_of_nodes)
        new_head_idx = torch.where(mask, head_idx, samples)
        new_tail_idx = torch.where(~mask, tail_idx, samples)
        # noinspection PyTypeChecker
        return new_head_idx, new_tail_idx


class BernoulliStrategy(SamplingStrategy):

    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph

    def sample(self, src_idx, node_idx):
        raise NotImplementedError()
