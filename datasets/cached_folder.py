import os
import numpy as np
from torch.utils.data import Dataset


class CachedFolder(Dataset):
    def __init__(self, root: str, subdir: str = None):
        self.root = os.path.expanduser(root)
        self.root = os.path.join(self.root, subdir) if subdir is not None else self.root
        self.files = list(sorted(f for f in os.listdir(self.root) if f.endswith('.npz')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        data = np.load(os.path.join(self.root, self.files[index]))
        if np.random.random() < 0.5:
            h, quant, idx = data['h'], data['quant'], data['idx']
        else:
            h, quant, idx = data['h_flip'], data['quant_flip'], data['idx_flip']
        return h, quant, idx
