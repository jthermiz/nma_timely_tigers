import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class TwoLayer(nn.Module):
    """Simple two layer MLP net"""

    def __init__(self, D_in, H, D_out):
        """Initialize class

        Parameters
        ----------
        D_in : int
            Number of input neurons
        H : int
            Number of hidden neurons
        D_out : int
            Number of output neurons
        """
        super(TwoLayer, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        self.lil = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, D_out),
        )

    def forward(self, x):
        """Compute forward pass of net

        Parameters
        ----------
        x : Tensor
            Batch of examples

        Returns
        -------
        Tensor
            Output of net
        """
        xf = self.lil(x)
        xf = self.softmax(xf)
        return xf


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, b, h, t, k):
        """
        Compute dot products. This is the same operation for each head,
        so we can fold the heads into the batch dimension and use torch.bmm
        Note: .contiguous() doesn't change the actual shape of the data,
        but it rearranges the tensor in memory, which will help speed up the computation
        for this batch matrix multiplication.
        .transpose(dim0, dim1) is used to change the shape of a tensor. It returns a new tensor
        that shares the data with the original tensor. It can only swap two dimension.
        Shape of `queries`: (`batch_size`, no. of queries, head,`k`)
        Shape of `keys`: (`batch_size`, no. of key-value pairs, head, `k`)
        Shape of `values`: (`batch_size`, no. of key-value pairs, head, value dimension)
        b: batch size
        h: number of heads
        t: number of keys/queries/values (for simplicity, let's assume they have the same sizes)
        k: embedding size
        """
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # Matrix Multiplication between the keys and queries
        score = torch.bmm(queries, keys.transpose(1, 2)) / \
            math.sqrt(k)  # size: (b * h, t, t)
        # row-wise normalization of weights
        softmax_weights = F.softmax(score, dim=2)

        # Matrix Multiplication between the output of the key and queries multiplication and values.
        out = torch.bmm(self.dropout(softmax_weights), values).view(
            b, h, t, k)  # rearrange h and t dims
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return out


class SelfAttention(nn.Module):
    """Multi-head self attention layer
    Args:
      k (int): Size of attention embeddings
      heads (int): Number of attention heads
    Attributes:
      to_keys: Transforms input to k x k*heads key vectors
      to_queries: Transforms input to k x k*heads query vectors
      to_values: Transforms input to k x k*heads value vectors
      unify_heads: combines queries, keys and values to a single vector
    """

    def __init__(self, k, heads=8, dropout=0.1):
        super().__init__()
        self.k, self.heads = k, heads

        self.to_keys = nn.Linear(k, k * heads, bias=False)
        self.to_queries = nn.Linear(k, k * heads, bias=False)
        self.to_values = nn.Linear(k, k * heads, bias=False)
        self.unify_heads = nn.Linear(k * heads, k)

        self.attention = DotProductAttention(dropout)

    def forward(self, x):
        """Implements forward pass of self-attention layer
        Args:
          x (torch.Tensor): batch x t x k sized input
        """
        b, t, k = x.size()
        h = self.heads

        # We reshape the queries, keys and values so that each head has its own dimension
        queries = self.to_queries(x).view(b, t, h, k)
        keys = self.to_keys(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        out = self.attention(queries, keys, values, b, h, t, k)

        return self.unify_heads(out)


class TransformerBlock(nn.Module):
    """Transformer Block
    Args:
      k (int): Attention embedding size
      heads (int): number of self-attention heads
    Attributes:
      attention: Multi-head SelfAttention layer
      norm_1, norm_2: LayerNorms
      mlp: feedforward neural network
    """

    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm_1 = nn.LayerNorm(k)
        self.norm_2 = nn.LayerNorm(k)

        hidden_size = 2 * k  # This is a somewhat arbitrary choice
        self.mlp = nn.Sequential(
            nn.Linear(k, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, k))

    def forward(self, x):
        attended = self.attention(x)
        # Complete the input of the first Add & Normalize layer
        x = self.norm_1(attended + x)

        feedforward = self.mlp(x)
        # Complete the input of the second Add & Normalize layer
        x = self.norm_2(feedforward + x)

        return x


class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, nhead=1, depth=2):
        super(Transformer, self).__init__()
        transformer_blocks = []

        for i in range(depth):
            transformer_blocks.append(TransformerBlock(in_dim, nhead))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.classification_head = nn.Linear(in_dim, out_dim)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        
        #x = self.pos_enc(x)
        x = self.transformer_blocks(x)
        sequence_avg = x.mean(dim=1)
        x = self.classification_head(sequence_avg)
        x = self.softmax(x)
        #x = torch.nn.functional.log_softmax(x, dim=1)

        return x
