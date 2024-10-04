import dataclasses

import numpy as np
import torch
from torch import nn
import dataclasses
from dataset import SimDataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GridModel(torch.nn.Module):

    @staticmethod
    def _get_tensors(encoding_size: int, receptors: int, resolution: int=1, source_distance_multiplier: int=4, add_extra_layer: bool=True, depth: float = 1.0):
        """
        Given input settings, builds the 2D grid an extra needed data
        """
        current_idx = 0
        source_indices = []
        receptor_indices = []

        if add_extra_layer:
            current_idx += 1

        for receptor_to_add in range(receptors):
            if receptor_to_add % source_distance_multiplier == 0:
                source_indices.append(current_idx)

            receptor_indices.append(current_idx)
            current_idx += resolution

        if add_extra_layer:
            current_idx += 1

        depth_cells = int(depth * current_idx)
        grid = torch.randn(size=[encoding_size, depth_cells, current_idx], dtype=torch.float32)
        return grid, torch.tensor(receptor_indices, dtype=torch.long), torch.tensor(source_indices)

    def _prepare_grid(self, grid: torch.Tensor, encode_depth: bool, zero_dynamic_state: bool) -> torch.Tensor:
        if zero_dynamic_state:
            grid[self.static_encoding_size:, :, :] = 0.0

        if encode_depth:
            first_dim_values = torch.arange(grid.shape[1], dtype=torch.float32).unsqueeze(1) / grid.shape[1]
            repeated_values = first_dim_values.repeat(1, grid.shape[2])
            grid[0, :, :] = repeated_values

        return grid

    def __init__(self, static_encoding_size: int, dynamic_encoding_size, receptors: int, resolution: int = 1, source_distance_multiplier: int = 4,
                 add_extra_layer: bool = True, depth: float = 1.0, encode_depth: bool=False, zero_dynamic_state: bool=False):
        """
        Builds the grid model of our system. Returns:
            1. A pytorch tensor of the grid. It will be of shape [width voxels, height voxels, encoding size]
            2. A pytorch tensor of the indices of the sources. I.e, one can do grid[0, source_indices] to get all the source values
            3. A pytorch tensor of the indices of the receptors
        :param encoding_size: the size of the vectors that describes the voxel. Is both the static and the dynamic size
        :param receptors: the number of receptors
        :param resolution: the resolution of our grid. Minimally, each voxel is of the size of distance between sources. However, the higher the resolution, the more voxels will be between them.
        :param source_distance_multiplier: for simplicity, sources can only be on same spots of receptors.
        :param add_extra_layer: add extra layer
        :param depth: the ratio of depth in relation to width
        """
        super().__init__()
        self.encoding_size = static_encoding_size + dynamic_encoding_size
        self.static_encoding_size = static_encoding_size
        self.dynamic_encoding_size = dynamic_encoding_size
        self.encode_depth = encode_depth
        self.zero_dynamic_state = zero_dynamic_state

        grid, self.receptor_indices, self.source_indices = self._get_tensors(
            encoding_size=self.encoding_size, receptors=receptors, resolution=resolution, source_distance_multiplier=source_distance_multiplier,
            add_extra_layer=add_extra_layer, depth=depth
        )

        self.grid = self._prepare_grid(grid, encode_depth=encode_depth, zero_dynamic_state=zero_dynamic_state)
        grid.requires_grad = True

    def _init_grid_before_run(self):
        if self.zero_dynamic_state:
            self.grid[self.static_encoding_size:, :, :] = 0.0
        else:
            self.grid[self.static_encoding_size:, :, :,] = torch.randn([self.grid[0], self.grid[1], self.dynamic_encoding_size], device=self.grid.device, dtype=self.grid.dtype)

    def update_grid(self, new_dynamic_state: torch.Tensor):
        self.grid[self.static_encoding_size:, :, :] = new_dynamic_state

    def start_run(self, source_idx):
        with torch.no_grad():
            self._init_grid_before_run()
            source_location = self.source_indices[source_idx]
            self.grid[self.static_encoding_size, 0, :] = 0.0 # zero all places not at sources
            self.grid[self.static_encoding_size, 0, source_location] = 1.0

    def get_receptor_amplitude(self):
        return self.grid[self.static_encoding_size, 0, self.receptor_indices]

    def get_static_state(self):
        return self.grid[:self.static_encoding_size, :, :]

    def get_dynamic_state(self):
        return self.grid[self.static_encoding_size:, :, :]


@dataclasses.dataclass
class TrainParams:
    epochs: int
    timestamp_batches: int
    save_model_n_step: int
    lr: float = 1e-3
    homogeneity_weight: float = 0


def homogeneity_loss(dynamic_state):
    diff_h = torch.abs(dynamic_state[:, 1:, :] - dynamic_state[:, :-1, :])
    diff_w = torch.abs(dynamic_state[:, :, 1:] - dynamic_state[:, :, :-1])
    return torch.mean(diff_h) + torch.mean(diff_w)


class NCA:

    def __init__(self, grid_model: GridModel, network: nn.Module, output_path: Path):
        self.grid = grid_model
        self.network = network
        self.output_path = output_path

    def _train_step(self, dataset: SimDataset, params: TrainParams):
        for source in range(len(dataset)):
            source_data = dataset[source].to(device=device)
            self.grid.start_run(source_idx=source)
            for timestamp in range(source_data.shape[1]):
                loss = self.criterion(self.grid.get_receptor_amplitude(), source_data[:, timestamp])
                loss.backward()
                (params.homogeneity_weight * homogeneity_loss(self.grid.get_dynamic_state())).backward()
                if timestamp % params.timestamp_batches == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def _save_model(self, epoch: int):
        (state_dir := self.output_path / Path('states')).mkdir(exist_ok=True, parents=True)
        torch.save(self.network.state_dict(), str((state_dir / Path(f'network_{epoch}.pt').resolve())))
        torch.save(self.grid.state_dict(), str((state_dir / Path(f'grid_{epoch}.pt').resolve())))

    def _get_image(self):
        state = self.grid.get_static_state().detach().cpu().permute([1, 2, 0]).numpy()
        static_states = state.reshape([-1, state.shape[2]])
        projection = PCA(n_components=3).fit_transform(static_states)
        color = (projection - projection.min())
        color = color / color.max()
        image = color.reshape([state.shape[0], state.shape[1], 3])
        return (image * 255).astype(np.uint8)

    def _write_output(self, epoch: int):
        (output_dir := self.output_path / Path('output')).mkdir(exist_ok=True, parents=True)
        Image.fromarray(self._get_image()).save(fp=str((output_dir / Path(f'state_{epoch}.png')).resolve()))

    def train(self, dataset: SimDataset, params: TrainParams):
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=params.lr)
        for epoch in tqdm(range(params.epochs), 'training'):
            self._train_step(dataset, params=params)
            self._write_output(epoch=epoch)
            if params.save_model_n_step != 0 and epoch % params.save_model_n_step == 0:
                self._save_model(epoch=epoch)


if __name__ == '__main__':
    g = GridModel(static_encoding_size=3, dynamic_encoding_size=4, receptors=281, resolution=1, source_distance_multiplier=4, add_extra_layer=False, encode_depth=True, zero_dynamic_state=True)
    dataset = SimDataset()
    output = Path('./experiments/test')
    output.mkdir(exist_ok=True, parents=True)
    nca = NCA(grid_model=g, network=None, output_path=output)

    train_params = TrainParams(epochs=5, timestamp_batches=10, save_model_n_step=0)
    nca.train(dataset=dataset, params=train_params)




