import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from model.util import clones
from transformers.activations import get_activation

def self_attention(Q,K,V, mask=None) :
    K_T = torch.transpose(K,-2,-1)
    res_matmul = torch.matmul(Q,K_T)

    d_K = Q.size()[-1]
    res_scale = res_matmul/math.sqrt(d_K)

    if mask is not None :
        res_scale = res_scale.masked_fill(mask == 0, -1e20)
    
    res_softmax = F.softmax(res_scale, dim=-1)
    
    res_final_matmul = torch.matmul(res_softmax,V)

    return res_final_matmul, res_softmax


class MultiHeadAttention(nn.Module) :
    def __init__(self, head_num=8, d_model=512, dropout=0.1) :
        super(MultiHeadAttention, self).__init__()

        self.head_num = head_num
        self.d_model = d_model
        self.d_K = self.d_V = d_model//head_num

        # Linear 내부 W matrix, bias matrix -> learnable matrix
        # y = xW^T + b
        self.w_Q = nn.Linear(d_model, d_model)
        self.w_K = nn.Linear(d_model, d_model)
        self.w_V = nn.Linear(d_model, d_model)

        self.w_O = nn.Linear(d_model, d_model)

        self.self_attention = self_attention
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None) :
        if mask is not None :
            mask = mask.unsqueeze(1)
        
        batch_num = Q.size(0)
        # Q : seq_len * d_model -> QW_Q^T : seq_len * d_model -> view : seq_len * head_num * d_K -> transpose : head_num * seq_len * d_K
        Q = self.w_Q(Q).view(batch_num, -1, self.head_num, self.d_K).transpose(1, 2)
        K = self.w_K(K).view(batch_num, -1, self.head_num, self.d_K).transpose(1, 2)
        V = self.w_V(V).view(batch_num, -1, self.head_num, self.d_K).transpose(1, 2)

        # attention_result : head_num * seq_len * d_K -> transpose : seq_len * head_num * d_K -> view(concat) : seq_len * (head_num*d_K=d_model)
        attention_result, attention_score = self.self_attention(Q,K,V, mask)
        attention_result = attention_result.transpose(1,2).contiguous().view(batch_num, -1, self.head_num * self.d_K)
        # Concat(head1,head2,...)W_O^T
        return self.w_O(attention_result)

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model*4)
        self.w_2 = nn.Linear(d_model*4, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim =True)
        std = x.std(-1, keepdim=True)    

        return self.a_2 * (x-mean)/ (std + self.eps) + self.b_2
    
class ResidualConnection(nn.Module):
    def __init__(self, size, dropout):
        super(ResidualConnection,self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout((sublayer(self.norm(x))))
    

class Encoder(nn.Module) :
    # input -> multi-head attention -> add&norm -> feed forward -> add&norm
    def __init__(self, d_model, head_num, dropout) :
        super(Encoder,self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_1 = ResidualConnection(size=d_model, dropout=dropout)

        self.feed_forward = FeedForward(d_model=d_model)
        self.residual_2 = ResidualConnection(size=d_model, dropout=dropout)

    def forward(self, input_, mask) : 
        x = self.residual_1(input_, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_2(x, lambda x: self.feed_forward(x))
        return x