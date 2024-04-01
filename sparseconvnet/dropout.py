# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.autograd import Function
from torch.nn import Module
from .utils import *
from .sparseConvNetTensor import SparseConvNetTensor


class Dropout(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        self.p = p

    def forward(self, input):
        output = SparseConvNetTensor()
        i = input.features
        if self.training:
            m = i.new().resize_(1).expand_as(i).fill_(1 - self.p)
            output.features = i * torch.bernoulli(m)
        else:
            output.features = i * (1 - self.p)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

    def input_spatial_size(self, out_size):
        return out_size


class BatchwiseDropout(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        self.p = p

    def forward(self, input):
        output = SparseConvNetTensor()
        i = input.features
        if self.training:
            m = i.new().resize_(1).expand(1, i.shape[1]).fill_(1 - self.p)
            output.features = i * torch.bernoulli(m)
        else:
            output.features = i * (1 - self.p)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

    def input_spatial_size(self, out_size):
        return out_size


class Dropout2d(Module):
    def __init__(self, p=0.5):
        Module.__init__(self)
        self.p = p

    def forward(self, input: SparseConvNetTensor):
        output = SparseConvNetTensor()
        i = input.features
        spatial_size = input.spatial_size
        assert len(spatial_size) == 2, "Dropout2d can only be used with 2D data"

        bs = input.batch_size()

        if self.training:
            num_features = i.shape[0]
            num_channels = i.shape[1]

            def dropout(i):
                i = i.view(
                    bs, spatial_size[0], spatial_size[1], num_channels
                )  # B, H, W, C
                i = i.permute(0, 3, 1, 2)  # B, C, H, W
                m = i.new().resize_(bs, num_channels, 1, 1).fill_(1 - self.p)
                o = i * torch.bernoulli(m)
                o = o.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
                o = o.view(-1, num_channels)  # B*H*W, C
                return o

            # Check if every location has data
            if num_features < bs * spatial_size[0] * spatial_size[1]:
                dummy = torch.zeros(
                    (bs * spatial_size[0] * spatial_size[1], num_channels)
                ).to(i.device)
                dummy[:num_features] = i
                o = dropout(dummy)
                output.features = o[:num_features]
            else:
                output.features = dropout(i)

        else:
            output.features = i * (1 - self.p)
        output.metadata = input.metadata
        output.spatial_size = spatial_size
        return output

    def input_spatial_size(self, out_size):
        return out_size
