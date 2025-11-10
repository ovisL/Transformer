import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from model.util import clones
from transformers.activations import get_activation

