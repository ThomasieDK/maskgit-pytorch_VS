import os
import numpy as np
from torch.utils.data import Dataset


class CachedFolder(Dataset):
    def __init__(self, root: str):
        self.root = os.path.expanduser(root)
        self.files = list(sorted(f for f in os.listdir(self.root) if f.endswith('.npz')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        """
        Returns:
            h: encoded latent vector, Tensor of shape (C, H, W)
            quant: quantized latent vector, Tensor of shape (C, H, W)
            idx: indices of quantized latent vector, Tensor of shape (H * W, )
            y: class label
        """
        data = np.load(os.path.join(self.root, self.files[index]), allow_pickle=True)

        if np.random.random() < 0.5:
            h, quant, idx = data['h'], data['quant'], data['idx']
        else:
            h, quant, idx = data['h_flip'], data['quant_flip'], data['idx_flip']

        if 'y' in data:
            y = data['y']
            return dict(y=y, h=h, quant=quant, idx=idx)
        else:
            return dict(h=h, quant=quant, idx=idx)
