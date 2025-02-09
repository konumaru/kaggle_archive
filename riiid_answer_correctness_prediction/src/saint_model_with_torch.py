import numpy as np

import torch
import torch.nn as nn


class EncoderEmbedding(nn.Module):
    def __init__(self, n_content, n_part, n_dims, seq_len, device):
        super(EncoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.device = device

        self.position_embed = nn.Embedding(seq_len, n_dims)
        self.content_embed = nn.Embedding(n_content, n_dims)
        self.part_embed = nn.Embedding(n_part, n_dims)

    def forward(self, content_id, part_id):
        seq = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        pos = self.position_embed(seq)

        content = self.content_embed(content_id)
        part = self.part_embed(part_id)
        return pos + content + part


class DecoderEmbedding(nn.Module):
    def __init__(self, n_response, n_dims, seq_len, device):
        super(DecoderEmbedding, self).__init__()
        self.n_dims = n_dims
        self.seq_len = seq_len
        self.device = device

        self.position_embed = nn.Embedding(seq_len, n_dims)
        self.response_embed = nn.Embedding(n_response, n_dims)

    def forward(self, response):
        seq = torch.arange(self.seq_len, device=self.device).unsqueeze(0)
        pos = self.position_embed(seq)

        res = self.response_embed(response)
        return pos + res


class SAINTModel(nn.Module):
    def __init__(
        self,
        n_questions,
        n_categories,
        n_responses,
        device="cpu",
        max_seq=100,
        d_model=512,
        encoder_dim=128,
        decoder_dim=128,
        num_heads=4,
    ):
        super().__init__()
        self.device = device
        self.encoder_embedding = EncoderEmbedding(
            n_content=n_questions,
            n_part=n_categories,
            n_dims=d_model,
            seq_len=max_seq,
            device=device,
        )
        self.decoder_embedding = DecoderEmbedding(
            n_response=n_responses,
            n_dims=d_model,
            seq_len=max_seq,
            device=device,
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
        )
        self.fc1 = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, q, c, r, src_pad_mask, tgt_pad_mask):
        enc = self.encoder_embedding(
            content_id=q,
            part_id=c,
        )
        dec = self.decoder_embedding(
            response=r,
        )
        x = self.transformer(
            enc,
            dec,
            src_key_padding_mask=src_pad_mask.T,
            tgt_key_padding_mask=tgt_pad_mask.T,
        )
        x = self.fc1(x)
        return x.squeeze(-1)
