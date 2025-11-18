from torch.nn import ModuleList
import torch.nn.functional as F
import copy
import math
import torch
import numpy as np

def clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
