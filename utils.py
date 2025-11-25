from torch.nn import ModuleList
import torch.nn.functional as F
import copy
import math
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import csv
from tqdm import tqdm

def clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def load_csv(file_path):
    print(f'Load Data | file path: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)

        lines = []
        for line in csv_reader:
            line[0] = line[0].replace(';',',')
            lines.append(line)
    print(f'Load Complete | file path: {file_path}')

    return lines

class TranslationDataset(Dataset) :
    def __init__(self, tokenizer, file_path, max_length) :
        pad_token_idx = tokenizer.pad_token_id
        csv_datas = load_csv(file_path)
        self.docs = []
        
        for line in tqdm(csv_datas):
            input_ = tokenizer.encode(line[0],max_length=max_length,truncation=True)
            rest = max_length - len(input_)
            input_ = torch.tensor(input_ + [pad_token_idx]*rest)

            target = tokenizer.encode(line[1], max_length=max_length, truncation=True)
            rest = max_length - len(target)
            target = torch.tensor(target+ [pad_token_idx] * rest)

            doc={
                'input_str': tokenizer.convert_ids_to_tokens(input_),
                'input':input_,                                        # input
                'input_mask': (input_ != pad_token_idx).unsqueeze(-2),       # input_mask
                'target_str': tokenizer.convert_ids_to_tokens(target),
                'target': target,                                       # target,
                'target_mask': self.make_std_mask(target, pad_token_idx),    # target_mask
                'token_num': (target[...,1:] != pad_token_idx).data.sum()  # token_num
            }
            self.docs.append(doc)

    @staticmethod
    def make_std_mask(target, pad_token_idx) :
        padding_mask = (target != pad_token_idx).unsqueeze(0) 
        seq_len = target.size(0)
        # torch.triu: 상삼각행렬 (대각선 위쪽을 1로 채움). diagonal=1은 대각선 바로 위부터 1.
        # 미래(위쪽)가 1(True)이 되므로 == 0을 해주면 현재와 과거만 True가 됨.
        no_future_mask = torch.triu(torch.ones((seq_len, seq_len), device=target.device), diagonal=1) == 0
        # 두 마스크 결합 (AND 연산)
        # (1, seq_len) & (seq_len, seq_len) -> (seq_len, seq_len)
        target_mask = padding_mask & no_future_mask

        return target_mask
    
    def __len__(self) :
       return len(self.docs)
    
    def __getitem__(self, index):
       item = self.docs[index]
       return item