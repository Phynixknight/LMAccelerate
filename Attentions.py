import torch
from torch import nn, Tensor
import torch.nn.functional as F
import time
import math

class Attention(nn.Module):
    def __init__(self, word_size:int=512, embed_dim:int=64, flash:bool=False) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.flash = flash
#         self.dim_K = torch.tensor(embed_dim)
        self.query = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.key  = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def self_attention(self, Q:Tensor, K:Tensor, V:Tensor) -> Tensor:
#         K_T = torch.transpose(K, 0, 1)
#         score = torch.matmul(Q, K_T)  / torch.sqrt(self.dim_K)
#         score = torch.softmax(score, dim=-1)
#         Z = torch.matmul(score, V)
        if self.flash:
            Z = F.scaled_dot_product_attention(Q, K, V)
        else:
            d_k = Q.size(-1)
            score = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
            score = torch.softmax(score, dim=-1)
            Z = score @ V
        return Z

    def forward(self, x:Tensor) -> Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Z = self.self_attention(Q, K, V)
        return Z


class MultiheadAttention(nn.Module):
    r"""
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_head:int=8) -> None:
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
#         self.dim_K = torch.tensor(embed_dim)
        self.proj = nn.Linear(in_features=embed_dim * n_head, out_features=embed_dim)
#         nn.init.xavier_uniform_(self.proj)
        self.multihead = nn.ModuleList([
            Attention(word_size, embed_dim) for _ in range(n_head)
        ])

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.multihead], dim=-1)
        Z = self.proj(Z_s)
        return Z


class  MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_query:int=8) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        self.proj = nn.Linear(in_features=embed_dim * n_query, out_features=embed_dim)
#         nn.init.xavier_normal_(self.proj)
        delattr(self, 'query')
        self.queries = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
            for _ in range(n_query)
        ])
#         self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
#         self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            self.self_attention(query(x), K, V) for query in self.queries
        ], dim=-1)
        Z = self.proj(Z_s)
        return Z


class  GroupedQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/2305.13245.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 n_grouped: int = 4, n_query_each_group:int=2) -> None:
        super().__init__(word_size, embed_dim)
        delattr(self, 'query')
        delattr(self, 'key')
        delattr(self, 'value')

        self.grouped = nn.ModuleList([
            MultiQueryAttention(word_size, embed_dim, n_query=n_query_each_group)
            for _ in range(n_grouped)
        ])
        self.proj = nn.Linear(in_features=embed_dim * n_grouped, out_features=embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.grouped], dim=-1)
        Z = self.proj(Z_s)
        return Z
