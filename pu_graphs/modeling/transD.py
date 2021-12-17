import numpy as np
import torch.nn.init
from torch import nn


class TransD(nn.Module):
    def __init__(self, n_nodes: int, n_relations: int,
                 entities_embedding_dim: int, relation_embedding_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.ent_embedding_dim = entities_embedding_dim
        self.rel_embedding_dim = relation_embedding_dim
        self.head_embedding = nn.Embedding(num_embeddings=n_nodes,
                                             embedding_dim=entities_embedding_dim)
        self.head_embedding_p = nn.Embedding(num_embeddings=n_nodes,
                                             embedding_dim=entities_embedding_dim)
        self.tail_embedding = nn.Embedding(num_embeddings=n_nodes,
                                             embedding_dim=entities_embedding_dim)
        self.tail_embedding_p = nn.Embedding(num_embeddings=n_nodes,
                                             embedding_dim=entities_embedding_dim)
        torch.nn.init.xavier_uniform_(self.head_embedding.weight)
        torch.nn.init.xavier_uniform_(self.head_embedding_p.weight)
        torch.nn.init.xavier_uniform_(self.tail_embedding.weight)
        torch.nn.init.xavier_uniform_(self.tail_embedding_p.weight)

        self.relation_embedding = nn.Embedding(num_embeddings=n_relations,
                                                 embedding_dim=relation_embedding_dim)
        self.relation_embedding_p = nn.Embedding(num_embeddings=n_relations,
                                                 embedding_dim=relation_embedding_dim)
        torch.nn.init.xavier_uniform_(self.relation_embedding.weight)
        torch.nn.init.xavier_uniform_(self.relation_embedding_p.weight)

    def forward(self, head_indices, tail_indices, relation_indices):
        h_p = self.head_embedding_p(head_indices)
        t_p = self.tail_embedding_p(tail_indices)
        r_p = self.relation_embedding_p(relation_indices)

        h = self.head_embedding(head_indices)
        t = self.tail_embedding(tail_indices)
        r = self.relation_embedding(relation_indices)

        matrix_rh = r_p.unsqueeze(2) @ h_p.unsqueeze(1)
        matrix_rh = matrix_rh + torch.ones_like(matrix_rh)
        matrix_rt = r_p.unsqueeze(2) @ t_p.unsqueeze(1)
        matrix_rt = matrix_rt + torch.ones_like(matrix_rt)

        head_projection = (matrix_rh @ h.unsqueeze(2)).squeeze(2)
        tail_projection = (matrix_rt @ t.unsqueeze(2)).squeeze(2)


        scores = (head_projection + r - tail_projection).norm(p=2, dim=1)**2
        return scores