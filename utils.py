from torch.nn import ModuleList
import torch.nn.functional as F
import copy
import math
import torch
from torch.utils.data import Dataset
import numpy as np

def clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class TranslationDataset(Dataset) :
    def __init__(self, tokenizer, file_path, max_length) :
        