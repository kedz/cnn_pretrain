import torch
import numpy as np

def get_index_frequency(inputs, mask_value=None):

    indices, frequency = np.unique(inputs.numpy(), return_counts=True) 
    sorted_freqs = torch.LongTensor(int(indices.max()) + 1).fill_(0)
    for i, f in zip(indices, frequency):
        if mask_value is not None and i == mask_value:
            continue 
        sorted_freqs[i] = int(f)
    return sorted_freqs
