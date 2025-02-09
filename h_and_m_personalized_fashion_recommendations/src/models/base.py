import torch
import torch.nn as nn
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


class CNNHead(nn.Module):
    def __init__(self, input_dim, input_chanel, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(input_chanel, 32, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 8, kernel_size=1),
        )
        self.cnn_output_dim = 8 * self._calc_conv1d_size(input_dim, kernel_size=1)
        self.top = nn.Linear(self.cnn_output_dim, output_dim)

    def forward(self, x):
        x = self.layer(x)
        x = self.top(x.view(-1, self.cnn_output_dim))
        return x

    def _calc_conv1d_size(self, l_src, kernel_size, padding=0, stride=1):
        l_out = (l_src + 2 * padding - (kernel_size - 1) - 1) / stride + 1
        return int(l_out)


class HMModel(nn.Module):
    def __init__(
        self,
        transformer_num_hidden_layers: int = 4,
        transformer_num_attention_heads: int = 4,
        article_num_unique: int = len(articleId_index),
        article_embedding_size: int = 128,
        customer_feat_dim: int = 4,
        max_seq_len: int = 16,
    ):
        super(HMModel, self).__init__()
        self.max_seq_len = max_seq_len

        self.article_embeddings = nn.Embedding(
            article_num_unique + 1, article_embedding_size, padding_idx=0
        )
        self.channel_embeddings = nn.Embedding(3, 16, padding_idx=0)
        self.article_freq_embeddings = nn.Embedding(10, 16, padding_idx=0)
        self.token_embeddings = nn.Embedding(10, 16, padding_idx=0)

        self.embedding_size = article_embedding_size + 48
        self.embeddings_layernorm = nn.LayerNorm(self.embedding_size, eps=1e-12)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_size,
                nhead=transformer_num_attention_heads,
                dim_feedforward=512,
                dropout=0.0,
                layer_norm_eps=1e-5,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=transformer_num_hidden_layers,
        )
        self.head_inputs_dim = max_seq_len * self.embedding_size + customer_feat_dim
        self.head = NextArticlePredictionHead(input_size=self.head_inputs_dim)

    def forward(self, inputs):
        x = torch.cat(
            (
                self.article_embeddings(inputs["article_id_seq"]),
                self.channel_embeddings(inputs["channel_id_seq"]),
                self.article_freq_embeddings(inputs["article_id_freq_seq"]),
                self.token_embeddings(inputs["active_token_id"]),
            ),
            dim=2,
        )

        x = self.embeddings_layernorm(x)
        # NOTE: Below is bug, https://github.com/pytorch/pytorch/issues/24816
        x = self.encoder(x, src_key_padding_mask=inputs["mask"])
        x = x.masked_fill(torch.isnan(x), 0)
        x = x.view(-1, self.max_seq_len * self.embedding_size)
        x = torch.cat((x, inputs["customer_feat"]), dim=1)
        x = self.head(x)
        return x
