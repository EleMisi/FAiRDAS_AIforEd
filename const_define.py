import os
import random

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# Set seed function
def set_seed(seed=424242):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # print(f'Random seed {seed} has been set.')
    return seed
