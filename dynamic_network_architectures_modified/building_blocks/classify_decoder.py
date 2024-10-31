import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.attention import StackedConvBlocksWithAttention

from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.classify_conv_encoder import ClassifyConvEncoder



class ClassifyUNetDecoder(nn.Module):
    def __init__(self,
                 encoder: ClassifyConvEncoder,
                 num_classes: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 #n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 #deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class adds two fully connected layers to the classify encoder
        It does not use the skips of the encoder (no deep supervision)

        the encoder goes all the way to the bottleneck (bottom of the U), so that's where the decoder picks up
        """
        super().__init__()
        #self.deep_supervision = deep_supervision      ### not ncessary
        #self.encoder = encoder
        self.num_classes = num_classes
        #n_stages_encoder = len(encoder.output_channels)

        # we start at the bottom of the bottleneck

        # 3D final activation size is 4x4x6 = 96
        # 96 x 320 features = 30720 parameters (not counting bias)

        # 2D final activation size is 4x6 = 24
        # 24 x 512 features = 12288 parameters (not counting bias)

        if conv_op == nn.Conv2d:
            FC1start = 4 * 6 * 512
        elif conv_op == nn.Conv3d:
            FC1start = 4 * 4 * 6 * 320

        FC1end = 1000
        self.FC1 = nn.Linear(FC1start, FC1end)
        self.FC2 = nn.Linear(FC1end, 3)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        """
        x = torch.flatten(x, start_dim=1)     # flatten to 1 dimensional vector with len 30720
        x = self.FC1(x)
        x = self.dropout(x)

        x = nn.functional.leaky_relu(x, inplace=True)
        x = self.FC2(x)
        x = self.dropout(x)

        return x

    #def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
    #    skip_sizes = []
    #    for s in range(len(self.encoder.strides) - 1):
    #        skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
    #        input_size = skip_sizes[-1]
        # print(skip_sizes)

    #    assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
    #    output = np.int64(0)
    #    for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
    #        output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
    #        output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
    #        if self.deep_supervision or (s == (len(self.stages) - 1)):
    #            output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)

    #    return output
