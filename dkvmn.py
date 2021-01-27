import torch
import numpy as np
import random
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEBUG = False


class DKVMN(nn.Module):
    def __init__(self, n_question,n_pid, seqlen,
                 q_embed_dim, qa_embed_dim,
                 memory_size,
                 final_fc_dim=512, l2 = 1e-5):
        super().__init__()
        self.n_question = n_question
        self.seqlen = seqlen
        self.n_pid = n_pid
        self.l2 = l2
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = q_embed_dim
        self.memory_value_state_dim = qa_embed_dim
        self.final_fc_dim = final_fc_dim

        # Initialize Memory
        # mx.sym.Variable('init_memory_key_weight')
        self.init_memory_key = nn.Parameter(
            0.01*torch.randn(self.memory_size, self.memory_key_state_dim))
        # (self.memory_size, self.memory_value_state_dim)
        self.init_memory_value = nn.Parameter(
            0.1 * torch.randn(self.memory_size, self.memory_value_state_dim))

        self.memory = MEMORY(memory_size=self.memory_size, memory_key_state_dim=self.memory_key_state_dim,
                            memory_value_state_dim=self.memory_value_state_dim, qa_embed_dim=qa_embed_dim)

        self.q_embed = nn.Embedding(self.n_question+1,self.q_embed_dim)
        self.qa_embed = nn.Embedding(2*self.n_question+1, self.qa_embed_dim)
        if self.n_pid>0:
            self.q_embed_diff = nn.Embedding(self.n_question+1,self.q_embed_dim)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim)
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)

        self.input_fc_net = nn.Sequential(
            nn.Linear(self.q_embed_dim, 50), nn.Tanh())
        self.output_fc_net = nn.Sequential(
            nn.Linear(self.memory_value_state_dim+50,
                      self.final_fc_dim), nn.Tanh(),
            nn.Linear(self.final_fc_dim, 1)
        )
        self.reset()
    
    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid>0:
                torch.nn.init.constant_(p, 0.)
                
    def forward(self, q_data, qa_data, target, pid_data = None):
        batch_size = q_data.size(1)
        # mx.sym.broadcast_to(mx.sym.expand_dims(init_memory_value, axis=0),shape=(self.batch_size, self.memory_size, self.memory_value_state_dim))
        memory_value = self.init_memory_value[None, :, :].expand(
            batch_size, -1, -1)
        init_memory_key = self.init_memory_key
        self.seqlen = q_data.size(0)

        # embedding
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)
        if self.n_pid>0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            pid_embed_data = self.difficult_param(pid_data)

            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        mem = self.memory
        value_read_content_l = []
        input_embed_l = []
        for i in range(self.seqlen):
            # Attention
            q = q_embed_data[i]
            correlation_weight = mem.attention(q, init_memory_key)

            # Read Process
            # Shape (batch_size, memory_state_dim)
            read_content = mem.read(memory_value, correlation_weight)
            # save intermedium data
            value_read_content_l.append(read_content[None, :, :])
            input_embed_l.append(q[None, :, :])

            # Write Process
            qa = qa_embed_data[i]
            memory_value = mem.write(qa, memory_value, correlation_weight)

        all_read_value_content = torch.cat(value_read_content_l, dim=0)
        input_embed_content = torch.cat(input_embed_l, dim=0)

        input_embed_content = self.input_fc_net(input_embed_content)

        read_content_embed = torch.cat(
            [all_read_value_content, input_embed_content], dim=2)
        pred = self.output_fc_net(read_content_embed)  # Size (Seqlen, BS, 1)

        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = pred.reshape(-1)  # logit
        mask = labels != -1
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        out = loss(masked_preds, masked_labels)
        return out.sum()+c_reg_loss, m(preds), mask.sum()


class MEMORY(nn.Module):
    """
        Implementation of Dynamic Key Value Network for Memory Tracing
        ToDo:
    """

    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, qa_embed_dim):
        super().__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_key_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.qa_embed_dim = qa_embed_dim
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.erase_net = nn.Sequential(
            nn.Linear(self.qa_embed_dim,self.memory_value_state_dim), nn.Sigmoid() 
            )
        self.add_net = nn.Sequential(
            nn.Linear(self.qa_embed_dim,self.memory_value_state_dim), nn.Tanh() 
            )
    


    def attention(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))#BS, MS
        m = nn.LogSoftmax(dim = 1)
        log_correlation_weight = m(similarity_score) # Shape: (batch_size, memory_size)
        return log_correlation_weight.exp()


    def read(self, memory_value, read_weight):
        read_weight = torch.reshape(read_weight, shape=(-1,1,self.memory_size))
        read_content = torch.matmul(read_weight, memory_value)
        read_content = torch.reshape(read_content, # Shape (batch_size, 1, memory_state_dim)
                                 shape=(-1,self.memory_value_state_dim))
        return read_content  #(batch_size, memory_state_dim)

    def write(self, control_input, memory, write_weight):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """

        ## erase_signal  Shape (batch_size, memory_state_dim)
        erase_signal = self.erase_net(control_input)
        ## add_signal  Shape (batch_size, memory_state_dim)
        add_signal = self.add_net(control_input)
        ## erase_mult  Shape (batch_size, memory_size, memory_state_dim)

        erase_mult = 1 - torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                          torch.reshape(erase_signal, shape=(-1, 1, self.memory_value_state_dim)))

        aggre_add_signal = torch.matmul(torch.reshape(write_weight, shape=(-1, self.memory_size, 1)),
                                          torch.reshape(add_signal, shape=(-1, 1, self.memory_value_state_dim)))
        new_memory = memory * erase_mult + aggre_add_signal
        return new_memory






