from typing import Optional

import torch.nn.init
from torch import nn


class ComplEx(nn.Module):

    def __init__(self, n_nodes: int, n_relations: int, embedding_dim: int, max_norm: Optional[float] = None):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.node_embedding_real = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim, max_norm=max_norm)
        self.node_embedding_img = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim, max_norm=max_norm)
        torch.nn.init.xavier_uniform_(self.node_embedding_real.weight)
        torch.nn.init.xavier_uniform_(self.node_embedding_img.weight)
        self.relation_embedding_real = nn.Embedding(
            num_embeddings=n_relations, embedding_dim=embedding_dim, max_norm=max_norm
        )
        self.relation_embedding_img = nn.Embedding(
            num_embeddings=n_relations, embedding_dim=embedding_dim, max_norm=max_norm
        )
        torch.nn.init.xavier_uniform_(self.relation_embedding_real.weight)
        torch.nn.init.xavier_uniform_(self.relation_embedding_img.weight)

    def forward(self, head_indices, tail_indices, relation_indices):
        head_embeddings_real = self.node_embedding_real(head_indices)
        tail_embeddings_real = self.node_embedding_real(tail_indices)
        relation_embeddings_real = self.relation_embedding_real(relation_indices)
        head_embeddings_img = self.node_embedding_img(head_indices)
        tail_embeddings_img = self.node_embedding_img(tail_indices)
        relation_embeddings_img = self.relation_embedding_img(relation_indices)
        scores = torch.sum(relation_embeddings_real * head_embeddings_real * tail_embeddings_real, dim=-1) \
                 + torch.sum(relation_embeddings_real * head_embeddings_img * tail_embeddings_img, dim=-1) \
                 + torch.sum(relation_embeddings_img * head_embeddings_real * tail_embeddings_img, dim=-1) \
                 - torch.sum(relation_embeddings_img * head_embeddings_img * tail_embeddings_real, dim=-1)
        return scores
