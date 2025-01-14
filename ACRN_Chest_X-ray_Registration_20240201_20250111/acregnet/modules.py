import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(in_channels, out_channels, kernel_size, stride, padding, use_batchnorm)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ConvELUConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_batchnorm=False):
        super().__init__()

        self.block = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size, stride, padding, use_batchnorm),
            nn.ELU(),
            Conv(out_channels, out_channels, kernel_size, stride, padding, use_batchnorm)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class UpConvELUConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_batchnorm=False):
        super().__init__()

        self.up = UpConv(in_channels, out_channels, kernel_size, stride, padding, use_batchnorm)
        self.conv = ConvELUConv(out_channels, out_channels, kernel_size, stride, padding, use_batchnorm)

    def forward(self, x, y):
        x = self.up(x)
        x += y
        x = F.elu(x)
        x = self.conv(x)
        x = F.elu(x)
        return x


class SpatialTransformer(nn.Module):
    """ https://github.com/voxelmorph/voxelmorph """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.size = size
        self.mode = mode

        vectors = [torch.arange(0, s) for s in self.size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, x, flow):

        new_locs = self.grid + flow

        for i in range(len(self.size)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.size[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]

        return F.grid_sample(x, new_locs, mode=self.mode, align_corners=True)
