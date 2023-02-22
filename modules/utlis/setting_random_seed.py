import random
import numpy as np
import os
import torch
import torch.backends.cudnn


def set_random_seed(seed, deterministic=False, tf_on=False, torch_on=True):
    """Set up seed.
    Args:
        seed (int, required): Seed to be used.
        deterministic (bool, optional): Whether reproducible or not. Default to False.
        tf_on (bool, optional): Whether to use tf library. Default to False.
        torch_on: Whether to use torch library. Default to True.
    """
    random.seed(seed)
    # seed numpy as global RNG seed
    np.random.seed(seed)
    if tf_on:
        import tensorflow as tf
        # seed Keras
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    if torch_on:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
