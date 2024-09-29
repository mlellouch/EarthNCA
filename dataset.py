import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class SimDataset(Dataset):

    def __init__(self, data_path: Path=Path(__file__).parent / Path('./data/data_michael_2d_acoust.npy')):
        self.data = np.load(data_path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, source_idx):
        return self.data[source_idx]

    def get_sources(self) -> int:
        return self.data.shape[0]

    def get_receptors(self) -> int:
        return self.data.shape[1]


if __name__ == '__main__':
    d = Dataset()
    a = d[0]