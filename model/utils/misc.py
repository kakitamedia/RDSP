import numpy as np
from collections import OrderedDict


def str2bool(s):
    return s.lower() in ('true', '1')

def fix_model_state_dict(state_dict):
    # remove 'module.' of dataparallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def save_compressed_array(path, array):
    np.savez_compressed(path, array)


def load_compressed_array(path):
    array = np.load(path)
    return array['arr_0']