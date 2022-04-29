import math
import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, att_type, hidden_dim):
        super().__init__()
        if att_type == 'dot':
            self.att = self.dot_attention
        elif att_type == 'additive':
            self.add_param = nn.Parameter(data=torch.randn(hidden_dim, 1))
            self.add_weight = nn.Parameter(data=torch.randn(2*hidden_dim, hidden_dim))
            self.att = self.additive_attenition
        elif att_type == 'multiplicative':
            self.mult_weight = nn.Parameter(data=torch.randn(hidden_dim, hidden_dim))
            self.att = self.multiplicative_attention
        self.additional_param = nn.Parameter(data=torch.randn(1, hidden_dim))

    def forward(self, values, keys=None, queries=None, mask=None):
        '''values: [b, source_seq_len, h_dim]''' # queries from decoder, keys, values from encoder
        if keys == None:
            queries = self.additional_param
            queries = torch.repeat_interleave(queries.unsqueeze(0), values.shape[0], dim=0) # [b, target_seq_len, h_dim]
            keys = values
        score = self.att(keys, queries)    # [b, target_seq_len, source_seq_len]
        # print(score.shape, mask.shape)
        # score[mask] = -float('inf')
        score.masked_fill(mask, -float('inf'))
        e = F.softmax(score, dim=-1)    # [b, target_seq_len, source_seq_len]
        # print(e)
        o = torch.bmm(e, values)    #[b, target_seq_len, h_dim]
        ### dodaj projekcje na koniec
        return o

    def dot_attention(self, keys, queries):
        '''
        :param keys: [b, source_seq_len, h_dim]
        :param queries: [b, target_seq_len, h_dim]
        '''
        keys = keys.permute(0, 2, 1)    # [b, h_dim, source_seq_len]
        score = torch.bmm(queries, keys) / math.sqrt(keys.shape[2])
        return score

    def additive_attenition(self, queries, keys): # ???
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
