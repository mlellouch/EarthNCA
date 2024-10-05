import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SimDataset(Dataset):

    @staticmethod
    def _linear_normalize(data: torch.Tensor) -> torch.Tensor:
        min_val, max_val = data.min(), data.max()
        data = -1 + ((data - min_val) / (max_val - min_val)) * 2
        return data

    @staticmethod
    def _logarithmic_normalize(data: torch.Tensor) -> torch.Tensor:
        neg = data < 0
        data[~neg] = torch.log(data[~neg] + 1)
        data[neg] = -torch.log(-data[neg] + 1)
        return data

    def __init__(self, data_path: Path=Path(__file__).parent / Path('./data/data_michael_2d_acoust.npy'), linear_normalize=False, logarithmic_normalize=False):
        self.data = torch.tensor(np.load(data_path), dtype=torch.float32)
        if linear_normalize:
            self.data = self._linear_normalize(self.data)

        elif logarithmic_normalize:
            self.data = self._linear_normalize(self._logarithmic_normalize(self.data))

        self.data = self.data.to(device=device)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, source_idx) -> torch.Tensor:
        return self.data[source_idx]

    def get_sources(self) -> int:
        return self.data.shape[0]

    def get_receptors(self) -> int:
        return self.data.shape[1]


if __name__ == '__main__':
    d = Dataset()
    a = d[0]