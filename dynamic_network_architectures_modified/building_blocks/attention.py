from typing import Tuple, List, Union, Type

import numpy as np
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD


class AttentionGate(nn.Module):
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super(AttentionGate, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        stride = maybe_convert_scalar_to_list(conv_op, stride)

        self.stride = stride

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}

        attention_gate_ops = []

        if conv_op == nn.Conv2d:
            self.conv = conv_op(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=[1, 1],
                padding=0,
                dilation=1,
                bias=conv_bias,
            )
        elif conv_op == nn.Conv3d:
            self.conv = conv_op(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=[1, 1, 1],
                padding=0,
                dilation=1,
                bias=conv_bias,
            )

        self.attention_gate = nn.Sequential(nn.LeakyReLU(inplace=True),
                                            self.conv,
                                            nn.Sigmoid())

    def forward(self, x):
        return self.attention_gate(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.stride), "just give the image size without color/feature channels or " \
                                                    "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                    "Give input_size=(x, y(, z))!"
        output_size = [i // j for i, j in zip(input_size, self.stride)]  # we always do same padding
        return np.prod([self.output_channels, *output_size], dtype=np.int64)


class StackedConvBlocksWithAttention(nn.Module):
    def __init__(self,
                 num_convs: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):

        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )


        ### classification encoder
        self.classify_convs = nn.Sequential(
            ConvDropoutNormReLU(
                conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                ConvDropoutNormReLU(
                    conv_op, output_channels[i - 1], output_channels[i], kernel_size, 1, conv_bias, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        ###  attention GATE pseudocode
        #x = seg encoder + classify              ###sum = LeakyReLU(conv_seg + conv_classify)
        #x = torch.nn.LeakyReLU(x)
        #x = self.conv1x1(x)                          ### feat = conv(sum)    # 1 x 1 x 1 convolution
        #alpha = sigmoid(x)
        #return alpha * classify


        if conv_op == nn.Conv2d:
            self.attention_gate = AttentionGate(conv_op = conv_op,
                                                input_channels = output_channels[-1],
                                                output_channels = 1,
                                                kernel_size = [1, 1],
                                                stride = 1,
                                                conv_bias = conv_bias,
                                                )

        elif conv_op == nn.Conv3d:
            self.attention_gate = AttentionGate(conv_op = conv_op,
                                                input_channels = output_channels[-1],
                                                output_channels = 1,
                                                kernel_size = [1, 1, 1],
                                                stride = 1,
                                                conv_bias = conv_bias,
                                                )




        # self.output_channels is basically 'features_per_stage' in plans.json
        self.output_channels = output_channels[-1]     # takes the end of the stack
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        classify = self.classify_convs(x)
        alpha = self.attention_gate(self.convs(x) + classify)


        return (alpha * classify)                         ### elemente wise multiplication

        ### end attention GATE

        #classify = classify * alpha


    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output





class StackedResidualBlocksWithAttention(nn.Module):
    def __init__(self,
                 n_blocks: int,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16
                 ):

        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks

        if block == BasicBlockD:
            blocks = nn.Sequential(
                block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                      norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                      squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                        norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                        squeeze_excitation, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
            )

        ### classification encoder
            classify_blocks = nn.Sequential(
                            block(conv_op, input_channels, output_channels[0], kernel_size, initial_stride, conv_bias,
                                  norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                                  squeeze_excitation, squeeze_excitation_reduction_ratio),
                            *[block(conv_op, output_channels[n - 1], output_channels[n], kernel_size, 1, conv_bias, norm_op,
                                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, stochastic_depth_p,
                                    squeeze_excitation, squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)]
                        )

        else:
            blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                      initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)])

        ### classification encoder
            classify_blocks = nn.Sequential(
                block(conv_op, input_channels, bottleneck_channels[0], output_channels[0], kernel_size,
                      initial_stride, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                      nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation, squeeze_excitation_reduction_ratio),
                *[block(conv_op, output_channels[n - 1], bottleneck_channels[n], output_channels[n], kernel_size,
                        1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                        nonlin, nonlin_kwargs, stochastic_depth_p, squeeze_excitation,
                        squeeze_excitation_reduction_ratio) for n in range(1, n_blocks)])


        self.blocks = blocks
        self.classify_blocks = classify_blocks


        ###  attention GATE pseudocode
        #x = seg encoder + classify              ###sum = LeakyReLU(conv_seg + conv_classify)
        #x = torch.nn.LeakyReLU(x)
        #x = self.conv1x1(x)                          ### feat = conv(sum)    # 1 x 1 x 1 convolution
        #alpha = sigmoid(x)
        #return alpha * classify


        self.attention_gate = AttentionGate(conv_op = conv_op,
                                            input_channels = output_channels[-1],
                                            output_channels = 1,
                                            kernel_size = [1, 1, 1],
                                            stride = 1,
                                            conv_bias = conv_bias,
                                            )


        # self.output_channels is basically 'features_per_stage' in plans.json
        self.output_channels = output_channels[-1]     # takes the end of the stack
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x):
        classify = self.classify_blocks(x)
        alpha = self.attention_gate(self.blocks(x) + classify)


        return (alpha * classify)                         ### elemente wise multiplication

        ### end attention GATE

        #classify = classify * alpha


    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.compute_conv_feature_map_size(size_after_stride)
        return output
