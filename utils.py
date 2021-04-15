import numpy as np
import torch
import random
import os
import time
import datetime

from config import *

major_version = 2
minor_version = 0
version = 2

def init_everything(seed):
    seed_everything(seed)
    init_config()
    print_version()

def print_version():
    decription = (
        f"Version: {major_version}.{minor_version}.{version}\n"
        f"val_every: {GlobalConfig.val_every}\n"
        f"half_precision: {GlobalConfig.half_precision}\n"
        f"dtype: {GlobalConfig.dtype}\n"
    )
    print(decription)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))