import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from parsing import *
import random 
import os
import numpy as np 
import torch 

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
