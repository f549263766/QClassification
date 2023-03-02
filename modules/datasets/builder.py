import copy
import platform
import random
import numpy as np
import torch
from functools import partial
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

# resource constraints to avoid multi-process problems
if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    r_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = r_limit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

# Create a register of dataset、pipeline and sampler
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
SAMPLERS = Registry('sampler')
