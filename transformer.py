import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
# from model.util import clones
# from transformers.activations import get_activation

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
    
class Decoder(nn.Module) :
    def __init__(self, d_model, head_num, dropout) :
        super(Decoder, self).__init__()
        
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_1 = ResidualConnection(size=d_model, dropout=dropout)

        self.encoder_decoder_attention = MultiHeadAttention(d_model=d_model, head_num=head_num)
        self.residual_2 = ResidualConnection(size=d_model, dropout=dropout)

        self.feed_forward = FeedForward(d_model=d_model)
        self.residual_3 = ResidualConnection(size=d_model, dropout=dropout)

    def forward(self, target, encoder_output, target_mask, encoder_mask) : 
        x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x,x,x,target_mask))
        x = self.residual_2(x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_3(x, self.feed_forward)

        return x
class Embeddings(nn.Module) :
    def __init__(self, vocab_num, d_model) :
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_num, d_model)
        self.d_model = d_model
    
    def forward(self, x) :
        return self.emb(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module) :
    def __init__(self, max_seq_len, d_model, dropout=0.1) :
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        # 홀수 ,짝수 인덱스 각각을 계산하기 위해 d_model//2
        base = torch.ones(d_model//2).fill_(10000)
        
        pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model,dtype=torch.float32)
        div_term = torch.pow(base,pow_term)
        
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        
        pe = pe.unsqueeze(0)
        # learnable은 아니지만 모델 저장, gpu 옮길 때 자동으로 같이 따라가도록 함
        # self.pe 대신 사용, 고정된 값
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        x = x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
    
# ### positionEncoding 시각화
# import matplotlib.pyplot as plt
# max_seq_len = 100   # 문장 길이 (Y축)
# d_model = 128       # 임베딩 차원 (X축, 시각화를 위해 128로 설정)
# # (내부적으로 __init__이 실행되면서 pe 행렬이 만들어짐
# pe_layer = PositionalEncoding(max_seq_len, d_model)
# pe_matrix = pe_layer.positional_encoding.squeeze(0).numpy()
# plt.figure(figsize=(15, 8))

# plt.subplot(1, 2, 1)
# plt.pcolormesh(pe_matrix, cmap='RdBu', shading='auto')
# plt.ylabel('Position (Sentence Index)')
# plt.xlabel('Dimension (d_model Index)')
# plt.title('Positional Encoding Heatmap')
# plt.colorbar()
# plt.ylim(max_seq_len, 0) # Y축 상단이 0이 되도록 반전

# plt.subplot(1, 2, 2)
# plt.plot(pe_matrix[:, 0], label='Dim 0 (Sin, High Freq)', alpha=0.7)
# plt.plot(pe_matrix[:, 1], label='Dim 1 (Cos, High Freq)', alpha=0.7)
# plt.plot(pe_matrix[:, 32], label='Dim 32 (Mid Freq)', alpha=0.7)
# plt.plot(pe_matrix[:, 64], label='Dim 64 (Low Freq)', alpha=0.7)

# plt.title('Sine/Cosine Waves at Specific Dimensions')
# plt.xlabel('Position')
# plt.ylabel('Value')
# plt.legend(loc='upper right')
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

