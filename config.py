import torch

class GlobalConfig:
    target_size = 10
    target_columns = []
    learning_rate = 2e-4
    scheduler_batch_update = True
    val_every = False
    half_precision = True
    dtype=None

def init_config():
    if GlobalConfig.half_precision:
        GlobalConfig.dtype=torch.float16
    else:
        GlobalConfig.dtype=torch.float32

