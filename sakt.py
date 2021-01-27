import torch
from torch import nn
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = True


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class SAKT(nn.Module):
    def __init__(self, n_question, d_model, n_blocks, final_fc_dim,
                dropout, fc_layer, n_heads, d_ff,  m_type):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            is_pos_embed_cat: dummy here
            m_type: different types of attention block
            decay_type : different option for putting decay
        """
        self.m_type = m_type
        self.n_question = n_question
        self.dropout = dropout
        self.fc_layer = fc_layer
        embed_l = d_model
        # n_question+1 ,d_model
        self.q_embed = nn.Embedding(self.n_question+1, embed_l)
        self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
        # Decoder Object. It contains stack of attention block
        self.decoder = Decoder(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                               d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff, m_type=self.m_type)

        if self.fc_layer == 1:  # one layer
            self.out = nn.Sequential(
                nn.Linear(d_model + embed_l,
                          final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 1)
            )

        if self.fc_layer == 2:  # two layer
            self.out = nn.Sequential(
                nn.Linear(d_model + embed_l,
                          final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )

        if self.fc_layer == 3:  # three layer
            self.out = nn.Sequential(
                nn.Linear(d_model + embed_l,
                          final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(final_fc_dim, 256), nn.ReLU(
                ), nn.Dropout(self.dropout),
                nn.Linear(256, 256), nn.ReLU(), nn.Dropout(self.dropout),
                nn.Linear(256, 1)
            )

    def forward(self, q_data, qa_data, target, pid_data = None):
        # Batch First
        #e_outputs = self.encoder(src, src_mask)
        #qa_reg_loss = compute_ranking_loss(self)
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model
        qa_embed_data = self.qa_embed(qa_data)  # BS, seqlen, d_model

        # BS.seqlen,d_model
        # Pass to the decoder
        # output shape BS,seqlen,d_model or d_model//2
        d_output = self.decoder(q_embed_data, qa_embed_data)  # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  # logit
        mask = labels >= -0.9
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum(), m(preds), mask.sum()


class Decoder(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout,  m_type, kq_same =  0):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
            m_type: type of blocks

        """
        self.d_model = d_model
        self.m_type = m_type
        self.position_embedding = PositionalEmbedding(
            d_model)

        
        self.blocks = nn.ModuleList([
            BasicBlock(d_model=d_model, d_feature=d_model // n_heads,
                        d_ff=d_ff, dropout=dropout, n_heads=n_heads,  kq_same=kq_same)
            for _ in range(n_blocks)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        #########
        # target shape  bs, seqlen
        seqlen, batch_size = qa_embed_data.size(1), q_embed_data.size(0)

        # Get the postional embedding fixed vector
        position_embed_data = self.position_embedding(qa_embed_data).expand(
            batch_size, -1, -1)  # batch_size , seq_len, d_model/2
        # Now add or concatenate with the q's and qa's embedding
        q_pos_embed = q_embed_data
        qa_pos_embed = qa_embed_data+position_embed_data

        x = qa_pos_embed
        y = q_pos_embed

        flag_first = True
        for block in self.blocks:
            if flag_first:
                y = get_block_result(
                    block, mask=0, query=y, key=x, values=x)
                flag_first = False
            else:
                y = get_block_result(
                    block, mask=0, query=y, key=y, values=x)
                flag_first = True
        return y



def get_block_result(block, mask, query, key, values, apply_pos=True):
    """
    Input:
        block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
        mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
        query : Query. In transformer paper it is the input for both encoder and decoder
        key : Keys. In transformer paper it is the input for both encoder and decoder
        Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)
        apply_pos: If true, position wise feedforward net will be applied as the transformer paper
    Output:
        query: Input gets changed over the layer and returned.

    """

    seqlen, batch_size = query.size(1), query.size(0)
    nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)),
                          k=mask).astype('uint8')
    src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
    if mask == 0:  # If 0, zero-padding is needed.
        # Calls block.masked_attn_head.forward() method
        att = block.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=True)
    else:
        # Calls block.masked_attn_head.forward() method
        att = block.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=False)
    query = query + block.dropout((att))
    query = block.layer_norm1(query)
    if apply_pos:
        pos = block.position_wise_feed_forward(query)
        query = query + block.dropout((pos))
        query = block.layer_norm3(query)
    return query


class BasicBlock(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout)

        # Position wise feedforward Net
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        # Two layer norm layer and one droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        # Decay Weights
        # self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))

    def forward(self, q, k, v, mask=None, zero_pad=False):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None, zero_pad=False):
    """
    This is called by Multi-head atention object to find the values.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    scores.masked_fill_(mask == 0, -1e32)

    # If decay type uses softmax
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen
    # run logistic regression
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    # Original use dropout
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        #position = torch.arange(0, max_len).unsqueeze(1).float()
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() *
        #                     -(math.log(10000.0) / d_model))
        #pe[:, 0::2] = torch.sin(position * div_term)
        #pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


if __name__ == "__main__":
    pass
