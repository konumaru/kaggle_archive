import math

import torch.nn as nn
import torch.nn.functional as F
from utils import load_pickle

articleId_index = load_pickle("../data/working/article_id_map.pkl")


class NextArticlePredictionHead(nn.Module):
    """
    memo: transformers4rec では最終層をLogSoftmaxにしている
    https://github.com/NVIDIA-Merlin/Transformers4Rec/blob/80596e89977c24736ed5ff22b6fef43fdd6a02f9/transformers4rec/torch/model/prediction_task.py#L321-L387
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = len(articleId_index),  # num_article_ids,
    ):
        super(NextArticlePredictionHead, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.module = nn.Sequential(
            nn.LayerNorm(self.input_size),
            nn.BatchNorm1d(self.input_size),
            nn.Linear(self.input_size, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.module(x)
        return x


# =============================================================================
# Transformer
# =============================================================================
# ref: https://github.com/Whiax/BERT-Transformer-Pytorch/blob/main/train.py
def attention(q, k, v, mask=None, dropout=None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])

    # mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()

        #        self.q_linear = nn.Linear(out_dim, out_dim)
        #        self.k_linear = nn.Linear(out_dim, out_dim)
        #        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim * 3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)

    def forward(self, x, y=None, mask=None):
        # in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y

        qkv = self.linear(x)  # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, : self.out_dim]  # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim : self.out_dim * 2]  # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim * 2 :]  # BS * SEQ_LEN * EMBED_SIZE_L

        # break into n_heads
        q, k, v = [
            self.split_heads(t) for t in (q, k, v)
        ]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [
            t.transpose(1, 2) for t in (q, k, v)
        ]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD

        # n_heads => attention => merge the heads => mix information
        scores = attention(
            q, k, v, mask, self.dropout
        )  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = (
            scores.transpose(1, 2).contiguous().view(scores.shape[0], -1, self.out_dim)
        )  # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE

        return out


class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim: int, num_layers: int = 4):
        super().__init__()
        encoders = []
        for _ in range(num_layers):
            encoders += [EncoderLayer(2, embedding_dim, 256)]
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x, mask)
        return x
