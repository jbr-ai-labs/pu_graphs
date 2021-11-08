import torch.nn.init
from torch import nn


class DistMult(nn.Module):

    def __init__(self, n_nodes: int, embedding_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.embedding_dim = embedding_dim
        # TODO: add option to work with graphs, where heads and tail are different kind of entities
        self.node_embedding = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)
        # TODO: extend to multiple relations
        self.relation_vector = nn.Parameter(torch.Tensor(embedding_dim), requires_grad=True)
        # TODO: look how initialized in paper, experiment with different initializations
        # TODO: remove hard code
        torch.nn.init.uniform_(self.relation_vector, a=-0.01, b=0.01)

    def forward(self, head_indices, tail_indices):
        head_embeddings = self.node_embedding(head_indices)
        tail_embeddings = self.node_embedding(tail_indices)
        scores = torch.sum(
            head_embeddings * self.relation_vector * tail_embeddings, dim=-1
        )
        return scores
