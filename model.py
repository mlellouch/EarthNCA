import torch
from typing import Tuple

class GridModel:

    def __init__(self, encoding_size: int, sources: int, receptors: int, resolution: int=1, max_depth=0, receptor_distance = 50.0, source_distance_multipler: int=4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Builds the grid model of our system. Returns:
            1. A pytorch tensor of the grid. It will be of shape [width voxels, height voxels, encoding size]
            2. A pytorch tensor of the indices of the sources. I.e, one can do grid[source_indices] to get all the source values
            3. A pytorch tensor of the indices of the receptors
        :param encoding_size: the size of the vectors that describes the voxel. Is both the static and the dynamic size
        :param sources: the number of sources
        :param receptors: the number of receptors
        :param resolution: the resolution of our grid. Minimally, each voxel is of the size of distance between sources. However, the higher the resolution, the more voxels will be between them.
        :param max_depth: the maximum depth in meters. If not described, will be the same distance as the ground, i.e. to build a square
        :param receptor_distance: the distance between two neighbor receptors.
        :param source_distance_multipler: for simplicity, sources can only be on same spots of receptors.
        """

        pass
