import torch
from transformer import Transformer

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import math
import os
import time

class TranslationTrainer() :
    def __init__(self,
               dataset,
               tokenizer,
               model,
               max_len,
               device,
               model_name,
               checkpoint_path,
               batch_size,
               ):
        ...
