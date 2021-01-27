import torch
from torch import nn
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True



class DKT(nn.Module):
    def __init__(self, n_question,n_pid, embed_l, hidden_dim, final_fc_dim =512, dropout =0, l2 = 1e-5):
        super().__init__()
        """
        Input:
        """
        self.n_question = n_question
        self.n_pid = n_pid
        self.l2 = l2
        self.dropout = dropout
        self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(
            input_size=embed_l,
            hidden_size=self.hidden_dim
        )
        if self.n_pid>0:
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim,
                        final_fc_dim), nn.ReLU(),nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ),nn.Dropout(self.dropout),
            nn.Linear(256, self.n_question+1)
        )
        self.reset()
    
    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid>0:
                torch.nn.init.constant_(p, 0.)


    def forward(self, q_data, qa_data, target,pid_data= None):
        qa_embed_data = self.qa_embed(qa_data)#  seqlen, BS,   d_model
        if self.n_pid>0:
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            pid_embed_data = self.difficult_param(pid_data)
            qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        batch_size, sl = qa_data.size(1), qa_data.size(0)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        hidden_seq, _ = self.rnn(qa_embed_data, h_0)
        h = torch.cat([h_0, hidden_seq], dim = 0)[:-1, :, :]#T,BS,hidden_dim
        pred = self.out(h)  # Size (Seqlen, BS, n_question+1)
        # pred = pred.view(-1, self.n_question+1)
        # qs = q_data.view(-1)
        pred = pred.reshape(-1, self.n_question+1)
        qs = q_data.reshape(-1)
        pred = pred[torch.arange(batch_size*sl).to(device), qs]

        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = pred.reshape(-1)  # logit
        mask = labels != -1
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        out = loss(masked_preds, masked_labels)
        return out.sum()+c_reg_loss, m(preds), mask.sum()