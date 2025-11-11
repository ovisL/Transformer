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