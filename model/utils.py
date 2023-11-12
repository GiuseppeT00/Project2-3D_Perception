import time
import random
import torch
import numpy as np


def fix_seeds(seed: int = 3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
