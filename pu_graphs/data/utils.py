import typing as ty

import dgl
import torch
from torch import LongTensor

from . import keys


def get_split(graph: dgl.DGLGraph, split_key: str) -> dgl.DGLGraph:
    mask_key = f"{split_key}_mask" if split_key != "valid" else "val_mask"
    split_mask = graph.edata[mask_key]
    split_edges_index = torch.nonzero(split_mask, as_tuple=False).squeeze()

    split_graph = graph.edge_subgraph(split_edges_index, preserve_nodes=True)
    split_graph.edata["etype"] = graph.edata["etype"][split_edges_index]

    return split_graph


def transform_as_pos_neg(batch: ty.Dict[str, ty.Any]):
    n_positives = batch[keys.head_idx].size(0)
    n_unlabeled = batch[keys.neg_head_idx].size(0)
    new_fields = {
        keys.head_idx: torch.cat([batch[keys.head_idx], batch[keys.neg_head_idx]]),
        keys.tail_idx: torch.cat([batch[keys.tail_idx], batch[keys.neg_tail_idx]]),
        keys.rel_idx: batch[keys.rel_idx].expand([2, -1]).flatten(),
        keys.labels: torch.cat([torch.ones(n_positives), torch.zeros(n_unlabeled)]),
    }
    batch.update(new_fields)
    return batch


def get_edges_as_triplets(graph: dgl.DGLGraph) -> LongTensor:
    """

    :param graph:
    :return: shape [n_edges, 3]
    """
    head_idx, tail_idx = graph.edges()
    relation_idx = graph.edata["etype"]

    assert len(head_idx) == len(tail_idx)
    assert len(head_idx) == len(relation_idx)

    # noinspection PyTypeChecker
    return torch.tensor(list(zip(
        *(head_idx, tail_idx, relation_idx)
    )))
