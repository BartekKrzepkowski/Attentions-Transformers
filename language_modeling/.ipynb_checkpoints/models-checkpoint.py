import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_log_prob = F.log_softmax(y_pred, dim=-1)
        mask = (y_true != 0).float()
        # from given cord I choose index (given axis) wrt value in it
        loss = torch.gather(y_log_prob, index=y_true.unsqueeze(-1), axis=-1).squeeze(-1)
        entropy_cond = - (F.softmax(y_pred, dim=-1) * y_log_prob).sum(axis=-1)
        eps = entropy_cond.detach().clone() + loss.detach().clone()
        loss = - (loss * mask).sum(axis=0)
        return loss.mean(), eps


class LM_RNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size=128, rnn=nn.LSTM):
        super().__init__()
        self.embs = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = rnn(emb_dim, hidden_size)
        self.ffn = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, x_lengths=None):
        x = self.embs(x)    # [sec_len, b, emb_dim]
        x = pack_padded_sequence(input=x, lengths=x_lengths, batch_first=False)
        x, _ = self.rnn(x)
        x = x._replace(data=self.ffn(x.data))
        x, _ = pad_packed_sequence(x, batch_first=False)    # [sec_len, b, vocab_size]
        return x

    def evaluation(self, x, k=7):
        x = self.embs(x)
        x, s = self.rnn(x)
        x = self.ffn(x)     # [1, 1, vocab_size]
        idx = self.nucleaus_decoding(x[-1])    # [1, 1]
        idxs = [idx]
        for _ in range(k):
            o = self.embs(idx)
            o, s = self.rnn(o, s)
            o = self.ffn(o)
            idx = o[-1].argmax(dim=-1, keepdim=True)
            idxs.append(idx)

        idxs = torch.cat(idxs).squeeze().cpu().numpy()
        return idxs

    def greedy_decoding(self, logit):
        idx = logit.argmax(dim=-1, keepdim=True)
        return idx

    def ktop_decoding(self, logit, k=10, temp=1.):
        probs = F.softmax(logit / temp, dim=-1)
        values, idxs = probs.topk(k=k, dim=-1)
        values = values / values.sum()  # [1, k]
        # dostosowanie do multinomial
        idx = idxs[0][torch.multinomial(input=values[0], num_samples=1)].unsqueeze(0)   # [1,1]
        return idx

    def nucleaus_decoding(self, logit, p=0.5, temp=0.5):
        i = 0
        v_prob = 0
        probs = F.softmax(logit / temp, dim=-1)
        values, idxs = torch.sort(probs, dim=-1, descending=True)
        while v_prob < p:
            v_prob += values[0, i]
            i += 1
        values = values[:, :i] / values[:, :i].sum()
        idxs = idxs[:, :i]
        idx = idxs[0][torch.multinomial(input=values[0], num_samples=1)].unsqueeze(0)
        return idx

    def random_decoding(self, logit, temp=1.):
        probs = F.softmax(logit / temp, dim=-1)
        idx = torch.multinomial(input=probs[0], num_samples=1).unsqueeze(0)
        return idx

from attention import Block
class LM_Transformer_Custom(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size=128, rnn=nn.LSTM):
        super().__init__()
        self.embs = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.zeros(size=(60, 1, emb_dim)))
        self.att = Block(emb_dim=emb_dim, value_dim=emb_dim, query_dim=emb_dim,
                             hidden_dim=emb_dim, n_head=8, att_type='dot')
    
    def forward(self, x, x_lengths=None):
        embs = self.embs(x)    # [sec_len, b, emb_dim]
        pos_embs = self.pos_emb[:embs.shape[0], :, :]    # positional embeddings
        embs = embs + pos_embs
        x = embs.permute(1, 0, 2)  # [b, seq_len, emb_dim]
        x = torch.dropout(x, p=0.1, train=True)
        mask = self.create_mask(x, x_lengths)
        x = self.att(x, mask)  # [b, sec_len, emb_dim]
        x = x.permute(1, 0, 2)  # [sec_len, b, emb_dim]
        x = x @ self.embs.weight.T    # [sec_len, b, vocab_size]
        return x

    def create_mask(self, x, lengths):
        mask = torch.triu(torch.ones((x.shape[1], x.shape[1])), diagonal=1).to(x.device.type)
        mask = mask.bool()
        return mask

    def evaluation(self, x, k=7):
        x = self.embs(x)
        x, s = self.rnn(x)
        x = self.ffn(x)     # [1, 1, vocab_size]
        idx = self.nucleaus_decoding(x[-1])    # [1, 1]
        idxs = [idx]
        for _ in range(k):
            o = self.embs(idx)
            o, s = self.rnn(o, s)
            o = self.ffn(o)
            idx = o[-1].argmax(dim=-1, keepdim=True)
            idxs.append(idx)

        idxs = torch.cat(idxs).squeeze().cpu().numpy()
        return idxs

    def greedy_decoding(self, logit):
        idx = logit.argmax(dim=-1, keepdim=True)
        return idx

    def ktop_decoding(self, logit, k=10, temp=1.):
        probs = F.softmax(logit / temp, dim=-1)
        values, idxs = probs.topk(k=k, dim=-1)
        values = values / values.sum()  # [1, k]
        # dostosowanie do multinomial
        idx = idxs[0][torch.multinomial(input=values[0], num_samples=1)].unsqueeze(0)   # [1,1]
        return idx

    def nucleaus_decoding(self, logit, p=0.5, temp=0.5):
        i = 0
        v_prob = 0
        probs = F.softmax(logit / temp, dim=-1)
        values, idxs = torch.sort(probs, dim=-1, descending=True)
        while v_prob < p:
            v_prob += values[0, i]
            i += 1
        values = values[:, :i] / values[:, :i].sum()
        idxs = idxs[:, :i]
        idx = idxs[0][torch.multinomial(input=values[0], num_samples=1)].unsqueeze(0)
        return idx

    def random_decoding(self, logit, temp=1.):
        probs = F.softmax(logit / temp, dim=-1)
        idx = torch.multinomial(input=probs[0], num_samples=1).unsqueeze(0)
        return idx

