import numpy as np
import torch
import random
import os

major_version = 1
minor_version = 2
version = 1

def print_version():
    print(f'Version: {major_version}.{minor_version}.{version}')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True