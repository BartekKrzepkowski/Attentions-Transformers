import torch
import torch.nn as nn


class MnistFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32*32*1, 2500), nn.BatchNorm1d(2500), nn.ReLU(),
                                 nn.Linear(2500, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
                                 # nn.Linear(2000, 1500), nn.BatchNorm1d(1500), nn.ReLU(),
                                 # nn.Linear(1500, 1000), nn.BatchNorm1d(1000), nn.ReLU(),
                                 nn.Linear(1000, 500), nn.BatchNorm1d(500), nn.ReLU(),
                                 nn.Linear(500, 10))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.net(x)
        return x


from attention import Attention
import numpy as np


class Transformer_Custom(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embs = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(size=(60, 1, emb_dim)))

        self.value = nn.Linear(emb_dim, emb_dim // 2, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim // 2, bias=False)
        self.query = nn.Linear(emb_dim, emb_dim // 2, bias=False)

        self.att = Attention(att_type='dot', hidden_dim=emb_dim // 2)
        self.ffn1 = nn.Linear(emb_dim // 2, emb_dim, bias=False)
        self.ffn2 = nn.Linear(emb_dim, 1, bias=True)

    def forward(self, x, x_lengths=None):
        embs = self.embs(x)  # [sec_len, b, emb_dim], b = number of sentences
        pos_embs = self.pos_emb[:embs.shape[0]]  # positional embeddings
        embs = embs + pos_embs
        x = embs.permute(1, 0, 2)  # [b, seq_len, emb_dim]
        x = torch.dropout(x, p=0.1, train=True)
        
        v = self.value(x)  # [b, seq_len, emb_dim//2]
        k = self.key(x)
        q = self.query(x)
        
        mask = self.create_mask(v, x_lengths)
        x = self.att(v, k, q, mask)  # [b, sec_len, emb_dim//2]
        
        x = x.permute(1, 0, 2)  # [sec_len, b, emb_dim//2]
        x = torch.dropout(self.ffn1(x), p=0.1, train=True)
        x = x + embs  # [sec_len, b, emb_dim]
        x = self.att(x, x_lengths) # zredukuj zdanie do wektora uzględniając maskowanie
        x = x.mean(axis=0)
        x = self.ffn2(x)  # [sec_len, b, 1]
        return x.squeeze(-1)

    def create_mask(self, x, lengths):
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], diagonal=1).to(x.device.type)
        mask = mask.bool()
        return mask


class Transformer_MHA(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embs = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.normal(0, 1 / np.sqrt(emb_dim), size=(60, emb_dim)))
        self.proj1 = nn.Linear(emb_dim, emb_dim // 2, bias=False)
        self.proj2 = nn.Linear(emb_dim, emb_dim // 2, bias=False)
        self.proj3 = nn.Linear(emb_dim, emb_dim // 2, bias=False)
        self.att = nn.MultiheadAttention(emb_dim, num_heads=1)
        self.ffn1 = nn.Linear(emb_dim // 2, emb_dim, bias=False)
        self.ffn2 = nn.Linear(emb_dim, 1, bias=True)

    def forward(self, x, x_lengths=None):
        embs = self.embs(x)  # [sec_len, b, emb_dim]
        pos_embs = self.pos_emb[:embs.shape[0], :].unsqueeze(1)  # positional embeddings
        embs = embs + pos_embs
        x = embs.permute(1, 0, 2)  # [b, seq_len, emb_dim]
        # x = torch.dropout(x, p=0.1, train=True)
        # x1 = self.proj1(x)  # [b, seq_len, emb_dim//2]
        # x2 = self.proj2(x)
        # x3 = self.proj3(x)
        mask = self.create_mask(x, x_lengths)
        x = self.att(x, x, x, mask)[0]  # [b, sec_len, emb_dim]
        x = x.permute(1, 0, 2)  # [sec_len, b, emb_dim]
        # x = torch.dropout(x, p=0.1, train=True)
        x = x + embs  # [sec_len, b, emb_dim]
        x = x.mean(axis=0)
        x = self.ffn2(x)  # [sec_len, b, 1]
        return x.squeeze(-1)

    def create_mask(self, x, lengths):
        mask = torch.triu(torch.ones(x.shape[1], x.shape[1], diagonal=1).to(x.device.type)
        mask = mask.bool()
        return mask


class Transformer_Based(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embs = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.normal(0, 1 / np.sqrt(emb_dim), size=(60, emb_dim)))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.ffn1 = nn.Linear(emb_dim // 2, emb_dim, bias=False)
        self.ffn2 = nn.Linear(emb_dim, 1, bias=True)

    def forward(self, x, x_lengths=None):
        embs = self.embs(x)  # [sec_len, b, emb_dim]
        pos_embs = self.pos_emb[:embs.shape[0], :].unsqueeze(1)  # positional embeddings
        embs = embs + pos_embs
        x = embs.permute(1, 0, 2)  # [b, seq_len, emb_dim]
        x = torch.dropout(x, p=0.1, train=True)
        mask = self.create_mask(x, x_lengths)
        # mask = torch.repeat_interleave(mask, 2, dim=0)
        x = self.transformer(x, mask=mask)
        x = x.permute(1, 0, 2)  # [sec_len, b, emb_dim]
        x = torch.dropout(x, p=0.1, train=True)
        x = x.mean(axis=0)
        x = self.ffn2(x)  # [b, 1]
        return x.squeeze(-1)

    def create_mask(self, x, lengths):
        mask = torch.zeros((x.shape[0], x.shape[1], x.shape[1])).to(x.device.type)
        for i, j in enumerate(lengths):
            mask[i, :, j:] = 1
        mask = mask.bool()
        return mask
