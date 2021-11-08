import typing as ty
import dgl
import torch


def get_split(graph: dgl.DGLGraph, split_key: str) -> dgl.DGLGraph:
    split_mask = graph.edata[f"{split_key}_edge_mask"]
    split_edges_index = torch.nonzero(split_mask, as_tuple=False).squeeze()

    split_graph = graph.edge_subgraph(split_edges_index, preserve_nodes=True)
    split_graph.edata["etype"] = graph.edata["etype"][split_edges_index]

    return split_graph
