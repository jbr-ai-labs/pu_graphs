import abc
import random
import typing as ty

import dgl


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


class BernoulliStrategy(SamplingStrategy):

    def __init__(self, graph: dgl.DGLGraph):
        self.graph = graph

    def sample(self, src_idx, node_idx):
        raise NotImplementedError()
