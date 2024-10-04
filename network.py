import torch
from torch import nn
import torch.nn.functional as F


class Conv3DNetwork(nn.Module):

    def __init__(self, kernel_size: int, encoding_size: int, channels: int, mlp_depth: int, mlp_width: int, output_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(in_channels=1, out_channels=channels, kernel_size=(encoding_size, kernel_size, kernel_size), padding=(0, pad, pad))

        self.first_linear = nn.Linear(in_features=channels, out_features=mlp_width)
        self.linears = nn.ModuleList([nn.Linear(mlp_width, mlp_width) for _ in range(mlp_depth)])
        self.last_linear = nn.Linear(in_features=mlp_width, out_features=output_size)

    def forward(self, x):
        x = x[None, None, ...]
        x = self.conv(x)
        x = x[0, :, 0, :, :].permute(1, 2, 0)

        # now run on linear
        x = F.relu(self.first_linear(x))
        for linear in self.linears:
            x = F.relu(linear(x))
        x = self.last_linear(x)
        return x.permute(2, 0, 1)

if __name__ == '__main__':
    grid = torch.randn(size=[7, 281, 281])
    net = Conv3DNetwork(kernel_size=3, encoding_size=7, channels=128, mlp_depth=3, mlp_width=128, output_size=4)
    net(grid)


