import math
import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, emb_dim, value_dim, query_dim, hidden_dim, n_head, att_type, attn_pdrop=0.1, compress=False):
        super().__init__()
        self.n_head = n_head
        self.key_proj = nn.Linear(emb_dim, value_dim, bias=False)
        self.query_proj = nn.Linear(emb_dim, query_dim, bias=False)
        self.value_proj = nn.Linear(emb_dim, value_dim, bias=False)
        self.compress = compress
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.final_proj = nn.Linear(value_dim, emb_dim, bias=False)
        
        if att_type == 'dot':
            self.att = self.dot_attention
        elif att_type == 'additive':
            self.add_param = nn.Parameter(data=torch.randn(hidden_dim, 1))
            self.add_weight = nn.Parameter(data=torch.randn(2*hidden_dim, hidden_dim))
            self.att = self.additive_attenition
        elif att_type == 'multiplicative':
            self.mult_weight = nn.Parameter(data=torch.randn(hidden_dim, hidden_dim))
            self.att = self.multiplicative_attention

    def forward(self, x, mask=None):        
        B, T, C = x.size()
        if mask is None:
            mask = torch.triu(torch.ones((T, T)), diagonal=1).to(x.device.type)
            mask = mask.bool()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if self.compress:
            # może by to rozbudować o atencje?
            q = q.sum(axis=1, keepdim=True)
        score = self.att(q, k)    # (B, nh, *, T)
        score.masked_fill(mask, -float('inf'))
        e = F.softmax(score, dim=-1)    
        o = e @ v    # (B, nh, *, hs)
        o = o.transpose(1, 2).contiguous().view(B, o.size(2), C) # re-assemble all head outputs side by side
        o = self.final_proj(o)
        o = self.attn_drop(o)
        if self.compress:
            o = o.squeeze(1)
        return o

    def dot_attention(self, q, k):
        '''
        :param keys: (B, nh, T, hs)
        :param queries: (B, nh, *, hs)
        '''
        k = k.transpose(-2, -1)    # (B, nh, hs, T)
        score = q @ k / math.sqrt(k.shape[2])
        return score # (B, nh, *, T)

    def additive_attenition(self, q, k): # ???
        if queries.shape[1] != keys.shape[1]:
            queries = torch.repeat_interleave(queries, values.shape[1]//queries.shape[1], dim=1)
        r_score = torch.tanh(torch.cat([keys, queries], axis=-1) @ self.add_weight) # [b, seq_len, h_dim]
        score = r_score @ self.add_param
        return score

    def multiplicative_attention(self, queries, keys):
        '''
        :param keys: [b, seq_len, h_dim]
        :param queries: [b, *, h_dim]
        '''
        keys = keys.permute(0, 2, 1)  # [b, h_dim, seq_len]
        l_score = queries @ self.mult_weight    # [b, *, h_dim]
        score = torch.bmm(l_score, keys)  # [b, *, seq_len]
        return score
    
    
class Block(nn.Module):
    def __init__(self, emb_dim, value_dim, query_dim, hidden_dim, n_head, att_type, attn_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.attn = Attention(emb_dim, value_dim, query_dim, hidden_dim, n_head, att_type)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(attn_pdrop),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x
    
