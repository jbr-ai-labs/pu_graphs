from typing import Optional

import torch.nn.init
from torch import nn


class DistMult(nn.Module):

    def __init__(self, n_nodes: int, n_relations: int, embedding_dim: int, max_norm: Optional[float] = None):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.node_embedding = nn.Embedding(
            num_embeddings=n_nodes, embedding_dim=embedding_dim, max_norm=max_norm
        )
        torch.nn.init.xavier_uniform_(self.node_embedding.weight)
        self.relation_embedding = nn.Embedding(
            num_embeddings=n_relations, embedding_dim=embedding_dim, max_norm=max_norm
        )
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, head_indices, tail_indices, relation_indices):
        head_embeddings = self.node_embedding(head_indices)
        tail_embeddings = self.node_embedding(tail_indices)
        relation_embeddings = self.relation_embedding(relation_indices)
        scores = torch.sum(
            head_embeddings * relation_embeddings * tail_embeddings, dim=-1
        )
        return scores
