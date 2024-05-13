from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
from typing import Union
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from pyutils.general import logger
from torch import Tensor, nn
from torch.nn.modules.activation import Hardtanh
from torch.types import Device, _size
from torchonn.devices.mrr import MORRConfig_20um_MQ

from .layers import AllPassMORRCirculantConv2d, AllPassMORRCirculantLinear
from .morr_base import MORR_CLASS_BASE

__all__ = ["MORR_CLASS_CNN"]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int = 3,
        miniblock: int = 8,
        bias: bool = False,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        MORRConfig=None,
        trainable_morr_scale: bool = False,
        trainable_morr_bias: bool = False,
        device: Device = torch.device("cuda"),
        dropout_rate: float = 0.04
    ) -> None:
        super().__init__()
        self.conv = AllPassMORRCirculantConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            miniblock=miniblock,
            bias=bias,
            mode=mode,
            v_max=v_max,
            v_pi=v_pi,
            in_bit=in_bit,
            w_bit=w_bit,
            MORRConfig=MORRConfig,
            trainable_morr_scale=trainable_morr_scale,
            trainable_morr_bias=trainable_morr_bias,
            device=device,
        )

        self.bn = nn.BatchNorm2d(out_channel)

        self.activation = Hardtanh(-1, 1, inplace=True)
        self.dropout = nn.Dropout(dropout_rate)  # add dropout layer

    def forward(self, x: Tensor) -> Tensor:
        x = self.activation(self.bn(self.conv(x)))
        x = self.dropout(x)  # apply dropout
        return x
    #    return self.activation(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        miniblock: int = 8,
        bias: bool = False,
        mode: str = "weight",
        v_max: float = 4.36,  # 0-pi for clements, # 6.166 is v_2pi, 0-2pi for reck
        v_pi: float = 4.36,
        w_bit: int = 16,
        in_bit: int = 16,
        activation: bool = True,
        MORRConfig=None,
        trainable_morr_scale: bool = False,
        trainable_morr_bias: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.linear = AllPassMORRCirculantLinear(
            in_channel,
            out_channel,
            miniblock=miniblock,
            bias=bias,
            mode=mode,
            v_max=v_max,
            v_pi=v_pi,
            in_bit=in_bit,
            w_bit=w_bit,
            MORRConfig=MORRConfig,
            trainable_morr_scale=trainable_morr_scale,
            trainable_morr_bias=trainable_morr_bias,
            device=device,
        )

        self.activation = Hardtanh(-1, 1, inplace=True) if activation else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class MORR_CLASS_CNN(MORR_CLASS_BASE):
    """MORR VGG16 for classification. MORR array-based convolution with learnable nonlinearity"""

    _conv = AllPassMORRCirculantConv2d
    _linear = AllPassMORRCirculantLinear
    _conv_linear = (AllPassMORRCirculantLinear, AllPassMORRCirculantConv2d)

    def __init__(
        self,
        img_height: int,
        img_width: int,
        in_channels: int,
        num_classes: int,
        kernel_list: List[int] = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        #kernel_list: List[Union[int, str]] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        maxpool_kernel_list: List[str] = ['M', 'M', 'M', 'M', 'M'],
        kernel_size_list: List[int] = [3]*13,
        block_list: List[int] = [1]*13,
        stride_list: List[int] = [1]*13,
        padding_list: List[int] = [1]*13,
        pool_out_size: int = 7,
        hidden_list: List[int] = [4096, 4096],
        in_bit: int = 32,
        w_bit: int = 32,
        mode: str = "weight",
        v_max: float = 10.8,
        v_pi: float = 4.36,
        act_thres: float = 6,
        photodetect: bool = True,
        bias: bool = False,
        MORRConfig=MORRConfig_20um_MQ,
        trainable_morr_bias: bool = False,
        trainable_morr_scale: bool = False,
        device: Device = torch.device("cuda"),
    ) -> None:
        # ... rest of the code remains the same as MORR_CLASS_CNN ...
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kernel_list = kernel_list
        self.maxpool_kernel_list = maxpool_kernel_list
        self.kernel_size_list = kernel_size_list
        self.block_list = block_list
        self.hidden_list = hidden_list
        self.stride_list = stride_list
        self.padding_list = padding_list
        self.pool_out_size = pool_out_size
        self.in_bit = in_bit
        self.w_bit = w_bit

        self.mode = mode
        self.v_max = v_max
        self.v_pi = v_pi
        self.act_thres = act_thres

        self.photodetect = photodetect
        self.bias = bias

        self.MORRConfig = MORRConfig
        self.trainable_morr_bias = trainable_morr_bias
        self.trainable_morr_scale = trainable_morr_scale

        self.device = device

        self.build_layers()
        self.drop_masks = None

        self.reset_parameters()
        
    # def build_layers(self) -> None:
    #     self.features = OrderedDict()
    #     for idx, x in enumerate(self.kernel_list, 0):
    #         if x == 'M':
    #             layer_name = "pool" + str(idx + 1)
    #             self.features[layer_name] = nn.MaxPool2d(kernel_size=2, stride=2)
    #         else:
    #             layer_name = "conv" + str(idx + 1)
    #             in_channel = self.in_channels if (idx == 0 or self.kernel_list[idx - 1] == 'M') else self.kernel_list[idx - 1]
    #             self.features[layer_name] = ConvBlock(
    #                 in_channel,
    #                 x,
    #                 self.kernel_size_list[idx],
    #                 stride=self.stride_list[idx],
    #                 padding=self.padding_list[idx],
    #                 miniblock=self.block_list[idx],
    def build_layers(self) -> None:
        self.features = OrderedDict()
        conv_list_index = 0
        maxpool_list_index = 0
        for idx in range(len(self.kernel_list) + len(self.maxpool_kernel_list)):
            if idx in self.maxpool_kernel_list:
                layer_name = "pool" + str(maxpool_list_index + 1)
                self.features[layer_name] = nn.MaxPool2d(kernel_size=2, stride=2)
                maxpool_list_index += 1
            else:
                if conv_list_index < len(self.kernel_list):
                    layer_name = "conv" + str(conv_list_index + 1)
                    in_channel = int(self.in_channels) if (conv_list_index == 0 or idx in self.maxpool_kernel_list) else int(self.kernel_list[conv_list_index - 1])
                    self.features[layer_name] = ConvBlock(
                        in_channel,
                        self.kernel_list[conv_list_index],
                        self.kernel_size_list[conv_list_index],
                        stride=self.stride_list[conv_list_index],
                        padding=self.padding_list[conv_list_index],
                        miniblock=self.block_list[conv_list_index],  
                        bias=self.bias,
                        mode="weight",
                        v_max=self.v_max,
                        v_pi=self.v_pi,
                        in_bit=self.in_bit,
                        w_bit=self.w_bit,
                        MORRConfig=self.MORRConfig,
                        trainable_morr_scale=self.trainable_morr_scale,
                        trainable_morr_bias=self.trainable_morr_bias,
                        device=self.device,
                    )
                conv_list_index += 1
                
        self.features = nn.Sequential(self.features)
        
        if self.pool_out_size > 0:
            self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
            feature_size = self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
        else:
            self.pool2d = None
            img_height, img_width = self.img_height, self.img_width
            for layer in self.modules():
                if isinstance(layer, self._conv):
                    img_height, img_width = layer.get_output_dim(img_height, img_width)
            feature_size = img_height * img_width * self.kernel_list[-1]

        self.classifier = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx + 1)
            in_channel = feature_size if idx == 0 else self.hidden_list[idx - 1]
            out_channel = hidden_dim
            self.classifier[layer_name] = LinearBlock(
                in_channel,
                out_channel,
                miniblock=self.block_list[idx],
                bias=self.bias,
                mode="weight",
                v_max=self.v_max,
                v_pi=self.v_pi,
                in_bit=self.in_bit,
                w_bit=self.w_bit,
                activation=True,
                MORRConfig=self.MORRConfig,
                trainable_morr_scale=self.trainable_morr_scale,
                trainable_morr_bias=self.trainable_morr_bias,
                device=self.device,
            )

        layer_name = "fc" + str(len(self.hidden_list) + 1)
        self.classifier[layer_name] = LinearBlock(
            self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
            self.num_classes,
            miniblock=self.block_list[-1],
            bias=self.bias,
            mode="weight",
            v_max=self.v_max,
            v_pi=self.v_pi,
            in_bit=self.in_bit,
            w_bit=self.w_bit,
            activation=False,
            MORRConfig=self.MORRConfig,
            trainable_morr_scale=self.trainable_morr_scale,
            trainable_morr_bias=self.trainable_morr_bias,
            device=self.device,
        )
        self.classifier["softmax"] = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(self.classifier)


# class MORR_CLASS_CNN(MORR_CLASS_BASE):
#     """MORR CNN for classification (MORR-ONN). MORR array-based convolution with learnable nonlinearity [SqueezeLight, DATE'21]"""

#     _conv = AllPassMORRCirculantConv2d
#     _linear = AllPassMORRCirculantLinear
#     _conv_linear = (AllPassMORRCirculantLinear, AllPassMORRCirculantConv2d)

#     def __init__(
#         self,
#         img_height: int,
#         img_width: int,
#         in_channels: int,
#         num_classes: int,
#         kernel_list: List[int] = [16],
#         kernel_size_list: List[int] = [3],
#         block_list: List[int] = [4],
#         stride_list: List[int] = [1],
#         padding_list: List[int] = [1],
#         pool_out_size: int = 5,
#         hidden_list: List[int] = [32],
#         in_bit: int = 32,
#         w_bit: int = 32,
#         mode: str = "weight",
#         v_max: float = 10.8,
#         v_pi: float = 4.36,
#         act_thres: float = 6,
#         photodetect: bool = True,
#         bias: bool = False,
#         # morr configuartion
#         MORRConfig=MORRConfig_20um_MQ,
#         trainable_morr_bias: bool = False,
#         trainable_morr_scale: bool = False,
#         device: Device = torch.device("cuda"),
#     ) -> None:
#         super().__init__()
#         self.img_height = img_height
#         self.img_width = img_width
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.kernel_list = kernel_list
#         self.kernel_size_list = kernel_size_list
#         self.block_list = block_list
#         self.hidden_list = hidden_list
#         self.stride_list = stride_list
#         self.padding_list = padding_list
#         self.pool_out_size = pool_out_size
#         self.in_bit = in_bit
#         self.w_bit = w_bit

#         self.mode = mode
#         self.v_max = v_max
#         self.v_pi = v_pi
#         self.act_thres = act_thres

#         self.photodetect = photodetect
#         self.bias = bias

#         self.MORRConfig = MORRConfig
#         self.trainable_morr_bias = trainable_morr_bias
#         self.trainable_morr_scale = trainable_morr_scale

#         self.device = device

#         self.build_layers()
#         self.drop_masks = None

#         self.reset_parameters()

#     def build_layers(self) -> None:
#         self.features = OrderedDict()
#         for idx, out_channel in enumerate(self.kernel_list, 0):
#             layer_name = "conv" + str(idx + 1)
#             in_channel = self.in_channels if (idx == 0) else self.kernel_list[idx - 1]
#             self.features[layer_name] = ConvBlock(
#                 in_channel,
#                 out_channel,
#                 self.kernel_size_list[idx],
#                 stride=self.stride_list[idx],
#                 padding=self.padding_list[idx],
#                 miniblock=self.block_list[idx],
#                 bias=self.bias,
#                 mode="weight",
#                 v_max=self.v_max,
#                 v_pi=self.v_pi,
#                 in_bit=self.in_bit,
#                 w_bit=self.w_bit,
#                 MORRConfig=self.MORRConfig,
#                 trainable_morr_scale=self.trainable_morr_scale,
#                 trainable_morr_bias=self.trainable_morr_bias,
#                 device=self.device,
#             )
#         self.features = nn.Sequential(self.features)

#         if self.pool_out_size > 0:
#             self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)
#             feature_size = self.kernel_list[-1] * self.pool_out_size * self.pool_out_size
#         else:
#             self.pool2d = None
#             img_height, img_width = self.img_height, self.img_width
#             for layer in self.modules():
#                 if isinstance(layer, self._conv):
#                     img_height, img_width = layer.get_output_dim(img_height, img_width)
#             feature_size = img_height * img_width * self.kernel_list[-1]

#         self.classifier = OrderedDict()
#         for idx, hidden_dim in enumerate(self.hidden_list, 0):
#             layer_name = "fc" + str(idx + 1)
#             in_channel = feature_size if idx == 0 else self.hidden_list[idx - 1]
#             out_channel = hidden_dim
#             self.classifier[layer_name] = LinearBlock(
#                 in_channel,
#                 out_channel,
#                 miniblock=self.block_list[idx],
#                 bias=self.bias,
#                 mode="weight",
#                 v_max=self.v_max,
#                 v_pi=self.v_pi,
#                 in_bit=self.in_bit,
#                 w_bit=self.w_bit,
#                 activation=True,
#                 MORRConfig=self.MORRConfig,
#                 trainable_morr_scale=self.trainable_morr_scale,
#                 trainable_morr_bias=self.trainable_morr_bias,
#                 device=self.device,
#             )

#         layer_name = "fc" + str(len(self.hidden_list) + 1)
#         self.classifier[layer_name] = LinearBlock(
#             self.hidden_list[-1] if len(self.hidden_list) > 0 else feature_size,
#             self.num_classes,
#             miniblock=self.block_list[-1],
#             bias=self.bias,
#             mode="weight",
#             v_max=self.v_max,
#             v_pi=self.v_pi,
#             in_bit=self.in_bit,
#             w_bit=self.w_bit,
#             activation=False,
#             MORRConfig=self.MORRConfig,
#             trainable_morr_scale=self.trainable_morr_scale,
#             trainable_morr_bias=self.trainable_morr_bias,
#             device=self.device,
#         )
#         self.classifier = nn.Sequential(self.classifier)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        if self.pool2d is not None:
            x = self.pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
